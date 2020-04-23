from typing import *

from framework import *
from .map_problem import MapProblem


class CachedMapDistanceFinder:
    """
    This is a helper class, used to find distances in the map and cache distances that has already been calculated.
    Calculating a distance (between 2 junctions) in the map is performed by solving a `MapProblem` using a
     `GraphProblemSolver`. `CachedMapDistanceFinder` receives the solver to use in its c'tor.
    """

    def __init__(self, streets_map: StreetsMap, map_problem_solver: GraphProblemSolver):
        self.streets_map = streets_map
        self.map_problem_solver = map_problem_solver

        self._cache: Dict[Tuple[int, int], Optional[Cost]] = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key: Tuple[int, int], val: Optional[Cost]):
        self._cache[key] = val

    def _get_from_cache(self, key: Tuple[int, int]) -> Optional[Cost]:
        return self._cache.get(key)

    def _is_in_cache(self, key: Tuple[int, int]) -> bool:
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return key in self._cache

    def get_map_cost_between(self, src_junction: Junction, tgt_junction: Junction) -> Optional[Cost]:
        """
        TODO [Ex.xx]: Implement this method!
        If the distance for the given source & target junctions is already stored in the cache, just return it.
        If the distance has not been stored in the cache yet, create a `MapProblem` with the given source & target,
         solve this problem using the `self.map_problem_solver` (that is given in the c'tor), store the cost of
         the solution in the cache, and finally return the cost of the solution. If the solver has not found a
         solution (the `final_search_node` field of the solver's result is None), the returned value should also
         be None. Even in this case (no solution found), you also should use the cache (store None in the cache).
        Use `_is_in_cache()`, `_get_from_cache()` and `_insert_to_cache()` methods to access the cache. Do not
         access the `_cache` field directly.
        The cache key should include the source & target indices.
        """

        # raise NotImplementedError  # TODO: remove this line!

        cache_key = (src_junction.index, tgt_junction.index)
        if self._is_in_cache(cache_key):
            return self._get_from_cache(cache_key)
        map_problem = MapProblem(streets_map=self.streets_map,
                                 source_junction_id=src_junction.index,
                                 target_junction_id=tgt_junction.index)
        map_problem_result = self.map_problem_solver.solve_problem(map_problem)
        map_distance = map_problem_result.solution_cost if map_problem_result.is_solution_found else None
        self._insert_to_cache(cache_key, map_distance)
        return map_distance
