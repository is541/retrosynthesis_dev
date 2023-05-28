"""Code for analysis."""

import math
from syntheseus.search.graph.and_or import OrNode, AndOrGraph
from syntheseus.search.graph.message_passing import run_message_passing
from syntheseus.search.algorithms.best_first.retro_star import reaction_number_update


def calculate_min_synthesis_cost(graph: AndOrGraph) -> None:
    """
    For graphs solved by retro-star, populates the field
    "reaction_number" with the minimum cost to reach this node
    from purchasable nodes.

    This is simple because retro* already essentially performs this calculation,
    although it uses a cost estimate for leaf nodes rather than their true cost
    (which is typically infinity).
    By setting all the cost estimates to infinity the retro* updates will
    yield the true cost.

    NOTE: this modifies the graph in-place, so the reaction numbers will no longer
    match their settings when the algorithm was run.
    """

    # Set reaction number estimates to infinity
    for node in graph.nodes():
        if isinstance(node, OrNode):
            node.data["reaction_number_estimate"] = math.inf

    # Run reaction number updates
    run_message_passing(
        graph=graph,
        nodes=sorted(graph.nodes(), key=lambda node: node.depth, reverse=True),
        update_fns=[reaction_number_update],
        update_predecessors=True,
        update_successors=False,
    )
