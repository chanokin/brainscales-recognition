#include <cypress/cypress.hpp>

using namespace cypress;

int main(int argc, const char *argv[])
{
    // Print some usage information
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <SIMULATOR>" << std::endl;
        return 1;
    }

    // Create the network description and run it
    Network net = Network();
    net.add_population<SpikeSourceArray>("source", 1, {100.0, 200.0, 300.0});
    net.add_population<IfCondExp>("neuron", 4, {}, {"spikes"});
    net.add_connection("source", "neuron", Connector::all_to_all(0.16));
    net.run(argv[1]);

    // Print the results
    for (auto neuron: net.population<IfCondExp>("neuron")) {
        std::cout << "Spike times for neuron " << neuron.nid() << std::endl;
        std::cout << neuron.signals().get_spikes();
    }
    return 0;
}
