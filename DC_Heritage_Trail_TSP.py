import googlemaps, pandas as pd, numpy as np, networkx as nx, osmnx as ox
from pulp import *

class TSP():
    # create empty variables that will persist across functions in the class
    signs = None
    dm = None
    x = None
    variables_dict = None
    prob = None
    g = None
    solution = None
    distance = None

    def __init__(self):
        # read in the raw heritage trail signs data
        signsDF = pd.read_excel('Heritage_Trail_Signs_and_Plaque.xlsx', sheet_name='Heritage_Trail_Signs_and_Plaque')
        # keep only the id, latitude, and longitude of each heritage trail sign
        self.signs = signsDF[['OBJECTID', 'LATITUDE', 'LONGITUDE']]
        # create the empty dataframe of the distance matrix
        dm = self.signs.set_index('OBJECTID').T
        dm.reset_index(drop=True, inplace=True)
        dm = dm.iloc[0:0]
        newSignsDF = self.signs.set_index('OBJECTID')
        newSignsDF = newSignsDF[[]]
        self.dm = pd.concat([newSignsDF, dm])
        return

    # calculate the walking distance between all starting and ending signs
    def calcDistMatrix(locations, frame):
        # google maps api key
        gmaps = googlemaps.Client(key='KEY')
        # create a list of ending signs
        columnsList = list(frame)
        # for each starting sign
        for row in frame:
            # retrieve the latitude and longitude
            start = np.array(locations.loc[locations['OBJECTID'] == row, ['LATITUDE', 'LONGITUDE']].values)
            start = str(start.item(0)) + ',' + str(start.item(1))
            # for each ending sign
            for column in columnsList:
                # retrieve the latitude and longitude
                end = np.array(locations.loc[locations['OBJECTID'] == column, ['LATITUDE', 'LONGITUDE']].values)
                end = str(end.item(0)) + ',' + str(end.item(1))
                # calculate the walking distance between the starting sign and ending sign using the google maps api
                frame.iloc[row-1].loc[column] = self.gmaps.distance_matrix(origins=start, destinations=end, mode='walking')['rows'][0]['elements'][0]['distance']['value']
        # return the distance matrix for all starting and ending signs
        return frame

    # create the linear program
    def build_model(self):
        # initialize the problem
        prob = LpProblem('DC_Heritage_Trail_TSP', LpMinimize)
        # generate distances
        distMatrix = calcDistMatrix(self.signs, self.dm)
        distances = []
        for index, row in distMatrix.iterrows():
            r = distMatrix[index].tolist()
            distances.append(r)
        # generate dictionary of signs and their distances
        distances_dict = dict(((a, b), distances[a][b]) for a in self.signs.index for b in self.signs.index if a != b)
        # create the objective function
        x = LpVariable.dicts('x', distances_dict, 0, 1, LpBinary)
        self.x = x
        self.variables_dict = dict([(str(value), key) for key, value in x.items()])
        cost = lpSum([x[(i, j)] * distances_dict[(i, j)] for (i, j) in distances_dict])
        # add the objective function to the model
        prob += cost
        # add other constraints to the model
        for k in self.signs.index:
            # every site has exactly one inbound connection
            prob += lpSum([x[(i, k)] for i in self.signs.index if (i, k) in x]) == 1
            # every site has exactly one outbound connection
            prob += lpSum([x[(k, i)] for i in self.signs.index if (k, i) in x]) == 1
        self.distances_dict = distances_dict
        self.prob = prob

    # check for subtours and add subtour elimination constraints
    def add_subtour_constraints(self):
        # generate the graph from the current solution
        varsdict = {}
        for v in self.prob.variables():
            if v.varValue == 1:
                varsdict[v.name] = v.varValue
        g = nx.Graph()
        # add nodes to the graph
        for index, row in self.signs.iterrows():
            lat = row['LATITUDE']
            lng = row['LONGITUDE']
            g.add_node(index, pos=(lat, lng))
        # add edges according to solution
        for k in varsdict:
            tmp_node = self.variables_dict[k]
            g.add_edge(tmp_node[0], tmp_node[1])
        self.g = g
        # if the number of connected components is one, then the solution is optimal
        nr_connected_components = nx.number_connected_components(G)
        if nr_connected_components == 1:
            return True
        # otherwise we need to add the subtour elimination constraints
        components = nx.connected_components(G)
        for c in components:
            self.prob += lpSum([self.x[(a, b)] for a in c for b in c if a != b]) <= len(c) - 1

    # solve the problem and check to see if there are subtours, then add subtour elimination constraints and resolve until there are no more subtours
    def solve(self):
        while True:
            self.prob.solve(solvers.PULP_CBC_CMD())
            if self.add_subtour_constraints():
                break

    # output the shortest route and total distance required
    def extract_solution(self):
        g = self.g
        start_point = self.signs[self.signs['OBJECTID'] == 1].index
        cycle = nx.find_cycle(g, start_point)
        solution = pd.DataFrame(cycle, columns=['Start_ID', 'End_ID'])
        # change values from index locations to id
        solution += 1
        # merge latitude and longitude back onto solution
        solution = solution.merge(self.signs, how='left', left_on='Start_ID', right_on='OBJECTID')
        solution = solution.merge(self.signs, how='left', left_on='End_ID', right_on='OBJECTID', suffixes=('_START', '_END'))
        self.solution = solution
        self.distance = value(self.prob.objective)

    def map_solution(self):
        # get walking map of dc
        g = ox.graph_from_place('District of Columbia, USA', network_type='walk')
        path = []
        # get nearest node of each sign and shortest path between nodes using walking map of dc
        for index, row in self.solution.iterrows():
            origin_node = ox.get_nearest_node(g, (row['LATITUDE_START'], row['LONGITUDE_START']))
            destination_node = ox.get_nearest_node(g, (row['LATITUDE_END'], row['LONGITUDE_END']))
            route = nx.shortest_path(g, origin_node, destination_node)
            path.extend(route)
        # remove duplicate signs in route since the previous ending point is the current starting point
        path = list(dict.fromkeys(path))
        # create a map of the route
        route_map = ox.induce_subgraph(G, path)
        m = ox.plot_graph_folium(route_map)
        m.save('DC_Heritage_Trail_TSP.html')

tsp = TSP()
tsp.build_model()
tsp.solve()
tsp.extract_solution()
tsp.map_solution()
