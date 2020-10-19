class WorldGrid:
    
    def __init__(self, resolution, threshold):
        self._resolution = resolution
        self._threshold = threshold
        self._grid = set()
        self._pool = {}

    def discretize_point(self, wp):
        return (int(wp[0] / self._resolution), int(wp[1] / self._resolution))

    def to_world_point(self, dp):
        return ((dp[0] + .5) * self._resolution, (dp[1] + .5) * self._resolution)
        
        
    def add_point(self, wp):

        dp = self.discretize_point(wp)

        if dp in self._grid:
            return
        
        if dp in self._pool:
            self._pool[dp] += 1
        else:
            self._pool[dp] = 1

        if self._pool[dp] >= self._threshold:
            self._grid.add(dp)

    def is_obstructed(self, wp):
        return self.discretize_point(wp) in self._grid
    
    def clear(self):
        self._grid = set()
        self._pool = {}

    def get_corners(self):
        
        li = []
        
        for p in self._grid:
            wp = self.to_world_point(p)
            li.append([
                (wp[0] + self._resolution / 2, wp[1] + self._resolution / 2),
                (wp[0] + self._resolution / 2, wp[1] - self._resolution / 2),
                (wp[0] - self._resolution / 2, wp[1] - self._resolution / 2),
                (wp[0] - self._resolution / 2, wp[1] + self._resolution / 2)
            ])
            
        return li

    def get_obstructed_tiles(self, discrete=False):
        return [p for p in self._grid] if discrete else [self.to_world_point(p) for p in self._grid]
