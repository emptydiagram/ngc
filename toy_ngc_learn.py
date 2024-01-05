from typing import Any
import numpy
import torch

class SNode:
    def __init__(self, name, dim=1, beta=1.0, leak=0.0, zeta=1.0, act_fn='identity'):
        if act_fn != 'identity':
            raise NotImplementedError("Only identity activation function is supported.")

        self.name = name
        self.dim = dim
        self.beta = beta
        self.leak = leak
        self.zeta = zeta
        self.act_fn = lambda x: x
        self.comp = {
            'z': torch.zeros([1, dim]),
            'phi(z)': torch.zeros([1, dim]),
            'dz_td': torch.zeros([1, dim]),
            'dz_bu': torch.torch.zeros([1, dim])
        }

        self.incoming_cables = []

    def get_signal(self, comp_name):
        return self.comp[comp_name]

    def step(self, debug=False):
        # reset bottom-up/top-down compartments
        self.comp['dz_td'] *= 0.
        self.comp['dz_bu'] *= 0.

        # collect data from incoming cables
        for cable in self.incoming_cables:
            self.comp[cable.dest_comp] += cable.propagate()

        dzdt = self.leak * self.comp['z'] + self.comp['dz_td'] + self.comp['dz_bu']
        self.comp['z'] = self.zeta * self.comp['z'] + self.beta * dzdt
        self.comp['phi(z)'] = self.act_fn(self.comp['z'])

        if debug:
            print(f"[{self.name}]", end=" ")
            for comp_name, comp_val in self.comp.items():
                print(f"{comp_name}: {comp_val.item()}", end=", ")
            print()



    def wire_to(self, dest_node, src_comp, dest_comp, cable_kernel):
        if cable_kernel['type'] == 'dense':
            cable = DCable(self, dest_node, src_comp, dest_comp, cable_kernel['init_kernels']['W_init'])
        elif cable_kernel['type'] == 'simple':
            cable = SCable(self, dest_node, src_comp, dest_comp)

        dest_node.incoming_cables.append(cable)
        return cable

    def clamp(self, comp_name, value):
        self.comp[comp_name] = value

class ENode:
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim
        self.comp = {
            'z': torch.zeros([1, dim]),
            'phi(z)': torch.zeros([1, dim]),
            'pred_targ': torch.zeros([1, dim]),
            'pred_mu': torch.zeros([1, dim])
        }

        self.incoming_cables = []

    def get_signal(self, comp_name):
        return self.comp[comp_name]
    
    def step(self):
        self.comp['z'] = self.comp['pred_targ'] - self.comp['pred_mu']
        pass

    def wire_to(self, dest_node, src_comp, dest_comp, cable_kernel):
        if cable_kernel['type'] == 'dense':
            cable = DCable(self, dest_node, src_comp, dest_comp, cable_kernel['init_kernels']['W_init'])
        elif cable_kernel['type'] == 'simple':
            cable = SCable(self, dest_node, src_comp, dest_comp)

        dest_node.incoming_cables.append(cable)
        return cable


class Cable:
    def __init__(self, src_node, dest_node, src_comp, dest_comp):
        self.src_node = src_node
        self.dest_node = dest_node
        self.src_comp = src_comp
        self.dest_comp = dest_comp

    def propagate(self):
        pass

class SCable(Cable):
    def __init__(self, src_node, dest_node, src_comp, dest_comp, scalar=1.0):
        super().__init__(src_node, dest_node, src_comp, dest_comp)
        self.scalar = scalar

    def propagate(self):
        inp = self.src_node.get_signal(self.src_comp)
        return self.scalar * inp

class DCable(Cable):
    def __init__(self, src_node, dest_node, src_comp, dest_comp, W_init=None, b_init=None):
        super().__init__(src_node, dest_node, src_comp, dest_comp)

        if W_init is None:
            self.W = torch.zeros([src_node.dim, dest_node.dim])
        else:
            W_init_type = W_init[0]
            if W_init_type == 'diagonal':
                dim = W_init[1]
                self.W = torch.diag(torch.ones([dim]) * W_init[1])
            elif W_init_type == 'gaussian':
                stddev = W_init[1]
                self.W = torch.empty([src_node.dim, dest_node.dim]).normal_(mean=0, std=stddev)
            else:
                raise NotImplementedError("Only diagonal initialization is supported.")

        if b_init is None:
            self.b = torch.zeros([1, dest_node.dim])
        else:
            self.b = torch.ones([1, dest_node.dim]) * b_init

    def propagate(self):
        inp = self.src_node.get_signal(self.src_comp)
        out = torch.matmul(inp, self.W) + self.b
        return out



class NGCGraph:
    def __init__(self, K):
        self.K = K
        self.nodes = {}
        self.cycle = None

    def set_cycle(self, nodes):
        self.cycle = nodes
        self.nodes = {node.name: node for node in nodes}

    def get_node(self, name):
        return self.nodes[name]

    def settle(self, clamped_vars=None, readout_vars=None):
        if clamped_vars is None:
            clamped_vars = []
        if readout_vars is None:
            readout_vars = []

        for (var_name, comp_name, val) in clamped_vars:
            self.nodes[var_name].comp[comp_name] = val

        for i in range(self.K):
            for node in self.cycle:
                node.step(debug=True)

        readouts = []
        for (var_name, comp_name) in readout_vars:
            readouts.append((var_name, comp_name, self.nodes[var_name].get_signal(comp_name)))
        return readouts


class GNCN_PDH:
    def __init__(self):
        pass


# "Simulating an NGC Circuit with Sensory Data" from ngc-learn Lesson 1
def circuit1():
    a = SNode('a', dim=1, beta=1, leak=0.0, act_fn='identity')
    b = SNode('b', dim=1, beta=1, leak=0.0, act_fn='identity')
    c = SNode('c', dim=1, beta=1, leak=0.0, act_fn='identity')

    init_kernels = {"W_init": ("diagonal", 1)}
    dcable_cfg = {"type": "dense", "init_kernels": init_kernels}
    a_b = a.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)
    c_b = c.wire_to(b, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg)

    circuit = NGCGraph(K=5)
    circuit.set_cycle([a, c, b])

    a_val = torch.ones([1, circuit.get_node('a').dim])
    c_val = torch.ones([1, circuit.get_node('c').dim])

    readouts = circuit.settle(
        clamped_vars=[('a', 'z', a_val), ('c', 'z', c_val)],
        readout_vars=[('b', 'phi(z)')]
    )

    b_val = readouts[0][2]
    print(b_val)


if __name__ == "__main__":
    circuit1()
