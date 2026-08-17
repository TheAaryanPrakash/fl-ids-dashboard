"""
Phase B — Minimal OpenFlow 1.3 SDN controller (Ryu substitute).

WHY NOT RYU: Ryu does not run on any Python version installable on this
machine. It has been unmaintained since ~2021 and its packaging/dependency
chain is broken at multiple independent, unfixable layers:
  - `pip install ryu` fails on Python 3.14 two ways: its setup.py calls a
    `setuptools.easy_install` API removed from modern setuptools, and even
    past that, its setuptools chain needs stdlib `distutils`, which was
    removed entirely in Python 3.12+.
  - Building a dedicated Python 3.10 venv (via `uv`, no sudo needed) and
    forcing an old setuptools gets `ryu` to build, but the resolver lands on
    ryu==2.2 (~2015), which needs the legacy `oslo.config` namespace
    package.
  - Pinning old oslo.config hits a THIRD wall: that old oslo-config's own
    code uses `collections.Mapping`, removed from Python 3.10's stdlib.
  - Dropping to Python 3.9 (last version keeping that deprecated alias)
    reveals a fourth, terminal wall: ryu==2.2's own source
    (ryu/lib/stringify.py) contains literal Python 2 print-statement
    syntax (`print "CLS", cls`) — that release never ran on Python 3 at
    all.
  - Explicitly pinning the newer ryu==4.34 instead gets further (it builds
    and imports past oslo.config), but fails on
    `eventlet.wsgi.ALREADY_HANDLED`, which doesn't exist in any
    pip-installable eventlet release, including the `eventlet<0.31` pin the
    build instructions anticipated as the fix.
That's a hard external blockage, not a config problem — so this module
plays Ryu's role directly: a minimal OpenFlow 1.3 learning-switch
controller (the same "simple_switch_13" tutorial pattern the instructions
asked for), built on `python-openflow` (`pyof`, a pure-protocol codec
library with no Ryu-style app framework or eventlet dependency) for OF1.3
message (de)serialization, plus stdlib `socket`/`threading` for the network
I/O Ryu would otherwise have supplied. Same conceptual role, same protocol,
no unmaintained framework underneath.

No IDS/monitoring logic here — pure L2 learning-switch forwarding, per
Phase B's scope.

Run directly (no ryu-manager equivalent needed):
    python3 network/controller.py [--port 6653]
"""
import argparse
import socket
import sys
import threading

from pyof.v0x04.asynchronous.packet_in import PacketIn
from pyof.v0x04.common.action import ActionOutput, ControllerMaxLen
from pyof.v0x04.common.constants import OFP_NO_BUFFER
from pyof.v0x04.common.flow_instructions import InstructionApplyAction
from pyof.v0x04.common.flow_match import Match, MatchType, OxmOfbMatchField, OxmTLV
from pyof.v0x04.common.header import Header, Type
from pyof.v0x04.common.port import PortNo
from pyof.v0x04.controller2switch.features_reply import FeaturesReply
from pyof.v0x04.controller2switch.features_request import FeaturesRequest
from pyof.v0x04.controller2switch.flow_mod import FlowMod, FlowModCommand
from pyof.v0x04.controller2switch.packet_out import PacketOut
from pyof.v0x04.symmetric.echo_reply import EchoReply
from pyof.v0x04.symmetric.echo_request import EchoRequest
from pyof.v0x04.symmetric.hello import Hello

OFP_PORT = 6653
IN_PORT_FIELD = OxmOfbMatchField.OFPXMT_OFB_IN_PORT


def log(msg):
    print(f"[controller] {msg}", flush=True)


def read_exact(sock, n):
    """Read exactly n bytes from a TCP stream, or raise ConnectionError."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-message")
        buf += chunk
    return buf


def recv_message(sock):
    """Read one OpenFlow message: parse the 8-byte header, then the body."""
    header_bytes = read_exact(sock, 8)
    header = Header()
    header.unpack(header_bytes)
    body = read_exact(sock, header.length.value - 8) if header.length.value > 8 else b""
    return header, body


def send_message(sock, msg):
    sock.sendall(msg.pack())


def build_table_miss_flow_mod(xid):
    """Match-everything, priority 0, send-to-controller — makes any
    unmatched packet trigger a PacketIn so the controller can learn/forward
    it. Installed once per switch right after the features handshake."""
    return FlowMod(
        xid=xid,
        cookie=0, cookie_mask=0, table_id=0,
        command=FlowModCommand.OFPFC_ADD,
        idle_timeout=0, hard_timeout=0, priority=0,
        buffer_id=OFP_NO_BUFFER,
        out_port=PortNo.OFPP_ANY, out_group=PortNo.OFPP_ANY,
        flags=0,
        match=Match(match_type=MatchType.OFPMT_OXM, oxm_match_fields=[]),
        instructions=[InstructionApplyAction(actions=[
            ActionOutput(port=PortNo.OFPP_CONTROLLER, max_length=ControllerMaxLen.OFPCML_NO_BUFFER),
        ])],
    )


def build_directed_flow_mod(xid, in_port, eth_dst, out_port):
    """Exact match on (in_port, eth_dst) -> output out_port. Installed once
    a destination MAC is learned, so future packets to it skip the
    controller entirely."""
    return FlowMod(
        xid=xid,
        cookie=0, cookie_mask=0, table_id=0,
        command=FlowModCommand.OFPFC_ADD,
        idle_timeout=60, hard_timeout=0, priority=10,
        buffer_id=OFP_NO_BUFFER,
        out_port=PortNo.OFPP_ANY, out_group=PortNo.OFPP_ANY,
        flags=0,
        match=Match(match_type=MatchType.OFPMT_OXM, oxm_match_fields=[
            OxmTLV(oxm_field=OxmOfbMatchField.OFPXMT_OFB_IN_PORT, oxm_value=in_port.to_bytes(4, "big")),
            OxmTLV(oxm_field=OxmOfbMatchField.OFPXMT_OFB_ETH_DST, oxm_value=eth_dst),
        ]),
        instructions=[InstructionApplyAction(actions=[
            ActionOutput(port=out_port, max_length=0),
        ])],
    )


def build_packet_out(xid, in_port, out_port, data):
    return PacketOut(
        xid=xid, buffer_id=OFP_NO_BUFFER, in_port=in_port,
        actions=[ActionOutput(port=out_port, max_length=0)],
        data=data,
    )


def extract_in_port(match):
    for field in match.oxm_match_fields:
        if int(field.oxm_field) == int(IN_PORT_FIELD):
            return int.from_bytes(field.oxm_value, "big")
    raise ValueError("PacketIn match has no in_port OXM field")


class SwitchConnection:
    """Per-switch state: connection socket, xid counter, learned MAC table."""

    def __init__(self, sock, addr):
        self.sock = sock
        self.addr = addr
        self.dpid = None
        self.mac_table = {}  # eth (bytes) -> port (int)
        self._xid = 0

    def next_xid(self):
        self._xid += 1
        return self._xid

    @staticmethod
    def _mac_str(mac_bytes):
        return ":".join(f"{b:02x}" for b in mac_bytes)

    def handle_packet_in(self, body):
        pkt_in = PacketIn()
        pkt_in.unpack(body)

        in_port = extract_in_port(pkt_in.match)
        data = pkt_in.data.value if hasattr(pkt_in.data, "value") else bytes(pkt_in.data)
        eth_dst, eth_src = data[0:6], data[6:12]

        learned = eth_src not in self.mac_table
        self.mac_table[eth_src] = in_port
        if learned:
            log(f"{self.addr}: learned {self._mac_str(eth_src)} on port {in_port}")

        if eth_dst in self.mac_table:
            out_port = self.mac_table[eth_dst]
            log(f"{self.addr}: {self._mac_str(eth_src)} -> {self._mac_str(eth_dst)} "
                f"known on port {out_port}, installing directed flow + forwarding")
            send_message(self.sock, build_directed_flow_mod(
                self.next_xid(), in_port, eth_dst, out_port))
        else:
            out_port = PortNo.OFPP_FLOOD
            log(f"{self.addr}: {self._mac_str(eth_src)} -> {self._mac_str(eth_dst)} "
                f"unknown dst, flooding")

        send_message(self.sock, build_packet_out(self.next_xid(), in_port, out_port, data))

    def run(self):
        send_message(self.sock, Hello(xid=self.next_xid()))
        send_message(self.sock, FeaturesRequest(xid=self.next_xid()))

        while True:
            header, body = recv_message(self.sock)
            mtype = header.message_type

            if mtype == Type.OFPT_HELLO:
                continue
            if mtype == Type.OFPT_ECHO_REQUEST:
                req = EchoRequest()
                req.unpack(body)
                reply = EchoReply(xid=header.xid.value, data=req.data)
                send_message(self.sock, reply)
            elif mtype == Type.OFPT_FEATURES_REPLY:
                fr = FeaturesReply()
                fr.unpack(body)
                self.dpid = str(fr.datapath_id)
                log(f"{self.addr}: features reply, datapath_id={self.dpid}, "
                    f"n_tables={int(fr.n_tables)} — installing table-miss flow")
                send_message(self.sock, build_table_miss_flow_mod(self.next_xid()))
            elif mtype == Type.OFPT_PACKET_IN:
                self.handle_packet_in(body)
            else:
                log(f"{self.addr}: ignoring message type {mtype}")


def handle_connection(sock, addr):
    log(f"switch connected: {addr}")
    conn = SwitchConnection(sock, addr)
    try:
        conn.run()
    except ConnectionError as exc:
        log(f"{addr}: disconnected ({exc})")
    except Exception as exc:  # noqa: BLE001 - keep one bad switch from killing the server
        log(f"{addr}: error, dropping connection: {exc!r}")
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description="Minimal OpenFlow 1.3 learning-switch controller.")
    parser.add_argument("--port", type=int, default=OFP_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(8)
    log(f"listening on {args.host}:{args.port} (OpenFlow 1.3, L2 learning switch)")
    sys.stdout.flush()

    try:
        while True:
            sock, addr = server.accept()
            t = threading.Thread(target=handle_connection, args=(sock, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    main()
