"""
Phase B — Mininet topology: 1 server host + 5 FL client hosts on one
OVS switch, matching the 5 FL clients already partitioned in
data/client_0.csv .. data/client_4.csv.

IPs (private /24): server h1 = 10.0.0.1, client host h{i+2} = 10.0.0.{i+2}
for client id i in 0..4 (10.0.0.2 .. 10.0.0.6).

The switch is pinned to OpenFlow13 since network/controller.py only speaks
OF1.3 (it's a from-scratch controller, not Ryu — see controller.py's module
docstring for why).

Usage (interactive):
    sudo mn --custom network/topo.py --topo mytopo --controller=remote
"""
from mininet.topo import Topo

N_CLIENTS = 5


class FLTopo(Topo):
    def build(self, n_clients=N_CLIENTS):
        switch = self.addSwitch("s1", protocols="OpenFlow13")

        server = self.addHost("h1", ip="10.0.0.1/24")
        self.addLink(server, switch)

        for i in range(n_clients):
            host = self.addHost(f"h{i + 2}", ip=f"10.0.0.{i + 2}/24")
            self.addLink(host, switch)


topos = {"mytopo": FLTopo}
