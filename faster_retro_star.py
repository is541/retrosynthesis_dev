from __future__ import annotations

from collections.abc import Sequence

from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch
from syntheseus.search.graph.and_or import ANDOR_NODE, AndOrGraph, OrNode


class ReduceValueFunctionCallsRetroStar(RetroStarSearch):
    """
    More efficient version of Retro* which saves value function calls.
    The difference is that retro* calls the value function (i.e. reaction number estimator)
    for every leaf node whereas this algorithm assigns a placeholder value of 0 to every leaf node
    and only calls the value function if it visits that node a second time.
    This essentially leaves the behaviour of retro* unchanged, but saves value function calls.

    The reason this works is that retro* greedily expands nodes on the current lowest-cost route,
    using the value function (reaction number) estimate as the cost of the node.
    If a node is not visited with a value function estimate of 0,
    then it would definitely not be visited with a non-zero value function estimate.
    Therefore if a node is not visited with a placeholder value of 0,
    it doesn't really matter what the value function estimate is.
    """

    def setup(self, graph: AndOrGraph) -> None:
        # If there is only 1 node, "visit" it by setting its reaction number estimate to 0
        # and incrementing its visit count
        if len(graph) == 1:
            graph.root_node.num_visit += 1
            graph.root_node.data.setdefault("reaction_number_estimate", 0.0)

        return super().setup(graph)

    def visit_node(self, node: OrNode, graph: AndOrGraph) -> Sequence[ANDOR_NODE]:
        """
        If node.num_visit == 0 then evaluate the value function and return.
        Otherwise expand.
        """
        assert node.num_visit >= 0  # should not be negative
        node.num_visit += 1
        if node.num_visit == 1:
            # Evaluate value function and return.
            node.data["reaction_number_estimate"] = self.reaction_number_estimator(
                [node]
            )[0]
            return []
        else:
            return super().visit_node(node, graph)

    def _set_reaction_number_estimate(
        self, or_nodes: Sequence[OrNode], graph: AndOrGraph
    ) -> None:
        for node in or_nodes:
            node.data.setdefault("reaction_number_estimate", 0.0)
