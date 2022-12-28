# Copyright 2022. Jonghyeon Ko
#
# All scripts in this folder are distributed as a free software. 
# We extended a few functions for fundamental concepts of process mining introduced in the complementary repository for BINet [1].
# 
# 1. [Nolle, T., Seeliger, A., Mühlhäuser, M.: BINet: Multivariate Business Process Anomaly Detection Using Deep Learning, 2018](https://doi.org/10.1007/978-3-319-98648-7_16)
# ==============================================================================


import uuid
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from GreenBPM.dir import PLOT_DIR
from GreenBPM import Apply_Trace
from GreenBPM import ProcessMap
import seaborn as sns


start_symbol = '▶'
end_symbol = '■' 

microsoft_colors = sns.color_palette(['#01b8aa', '#374649', '#fd625e', '#f2c80f', '#5f6b6d',
                                      '#8ad4eb', '#fe9666', '#a66999', '#3599b8', '#dfbfbf', '#008631', '#98BF64'])

class EventLogGenerator(object):
    def __init__(self, process_map=None):
        self.process_map = None
        self.likelihood_graph = None
        if process_map is not None:
            if isinstance(process_map, str):
                self.process_map = ProcessMap.from_plg(process_map)
            elif isinstance(process_map, ProcessMap):
                self.process_map = process_map
            else:
                raise TypeError('Only String and ProcessMap are supported.')

    @staticmethod
    def build_likelihood_graph(self,
                               activity_dependency_p=0.0,
                               probability_variance_max=None,
                               seed=None):

        def add_activity_dependency_to(g, source):
            source_value = self.likelihood_graph.nodes[source]['value']

            if source_value ==  end_symbol:
                return
            else:
                targets = []
                for target in g.successors(source_value):
                    if target not in nodes:
                        nodes[target] = []

                    split_activity = np.random.uniform(0, 1) <= activity_dependency_p
                    if (split_activity or not nodes[target]) and target !=  end_symbol:
                        identifier = uuid.uuid1()
                        nodes[target].append(identifier)
                        self.likelihood_graph.add_node(identifier, value=target, name='name')
                        targets.append(identifier)
                    else:
                        targets.append(np.random.choice(nodes[target]))

                for target in targets:
                    if source_value !=  start_symbol:
                        if source not in edges:
                            edges[source] = []
                        if target not in edges[source]:
                            self.likelihood_graph.add_edge(source, target)
                            edges[source].append(target)
                    else:
                        self.likelihood_graph.add_edge(source, target)

                    add_activity_dependency_to(g, target)

        # Set seed for consistency
        if seed is not None:
            np.random.seed(seed)

        # Init graph
        self.likelihood_graph = nx.DiGraph()

        # Init helper dictionaries
        nodes = {}
        edges = {}
        for node in self.process_map.graph:
            if node in [ start_symbol,  end_symbol]:
                self.likelihood_graph.add_node(node, value=node, name='name')
                nodes[node] = [node]

        # Add attribute and activity dependencies
        add_activity_dependency_to(self.process_map.graph,  start_symbol)

        # Annotate with probabilities
        for node in self.likelihood_graph:
            if node ==  end_symbol:
                continue

            successors = list(self.likelihood_graph.successors(node))

            if probability_variance_max is not None:
                variance = np.random.random() * np.abs(probability_variance_max) + .0001
                probabilities = np.abs(np.random.normal(0, variance, len(successors)))
                probabilities /= np.sum(probabilities)
            else:
                probabilities = np.ones(len(successors)) / len(successors)

            for successor, probability in zip(successors, probabilities):
                self.likelihood_graph.nodes[successor]['probability'] = probability
                self.likelihood_graph.edges[node, successor]['probability'] = np.round(probability, 2)

        return self.likelihood_graph

    def generate(self,
                 size,
                 activity_dependency_p=.5,
                 probability_variance_max=None,
                 seed=None,
                 show_progress='tqdm',
                 likelihood_graph=None):

        def random_walk(g):
            node =  start_symbol

            # Random walk until we reach the end event
            path = []
            while node !=  end_symbol:
                # Skip the start node
                if node !=  start_symbol:
                    path.append(node)

                # Get successors for node
                successors = list(g.successors(node))

                # Retrieve probabilities from nodes
                p = [g.edges[node, s]['probability'] for s in successors]

                # Check for and fix rounding errors
                if np.sum(p) != 0:
                    p /= np.sum(p)

                # Chose random successor based on probabilities
                node = np.random.choice(successors, p=p)

            return path

        if seed is not None:
            np.random.seed(seed)

        # Build the likelihood graph
        if likelihood_graph is not None:
            self.likelihood_graph = likelihood_graph
        else:
            self.build_likelihood_graph(self,
                activity_dependency_p=activity_dependency_p,
                probability_variance_max=probability_variance_max,
                seed=seed
            )

        # Add metadata to anomalies
        activities = sorted(list(set([self.likelihood_graph.nodes[node]['value'] for node in self.likelihood_graph
                                      if self.likelihood_graph.nodes[node]['name'] == 'name'
                                      and self.likelihood_graph.nodes[node]['value'] not in
                                      [ start_symbol,  end_symbol]])))
        normal = Apply_Trace()
        normal.activities = activities
        normal.graph = self.likelihood_graph

        # Generate the event log
        if show_progress == 'tqdm':
            from tqdm import tqdm
            iter = tqdm(range(size), desc='Generate event log')
        elif show_progress == 'tqdm_notebook':
            from tqdm import tqdm_notebook
            iter = tqdm_notebook(range(size), desc='Generate event log')
        else:
            iter = range(size)

        # Apply anomalies and add case id
        event_log = pd.DataFrame({'Case_ID': [], 'Activity': [], 'Order': [] })
        for case_id, path in enumerate([random_walk(self.likelihood_graph) for _ in iter], start=1):
            trace = normal.apply_to_path(path)
            df_case= pd.DataFrame({'Activity' : [ trace.events[i].name for i in range(0, len(trace.events))] })
            df_case['Case_ID'] = case_id
            df_case['Order'] = [ i for i in range(0, len(trace.events)) ]
            event_log = (pd.concat([event_log, df_case]))
        return event_log

    def plot_likelihood_graph(self, file_name=None, figsize=None):
        from matplotlib import pylab as plt

        # l = self.likelihood_graph
        l = self.likelihood_graph
        pos = nx.drawing.nx_agraph.graphviz_layout(l, prog='dot')

        if figsize is None:
            figsize = (10, 14)
        fig = plt.figure(1, figsize=figsize)
        
        color_map = []
        for node in l:
            if node in [ start_symbol,  end_symbol]:
                color_map.append(microsoft_colors[0])
            else:
                color_map.append(microsoft_colors[2])
                
        nx.draw(l, pos, node_color=color_map)
        nx.draw_networkx_labels(l, pos, labels=nx.get_node_attributes(l, 'value'))
        nx.draw_networkx_edge_labels(l, pos, edge_labels=nx.get_edge_attributes(l, 'probability'))

        if file_name is not None:
            # Save to disk
            fig.savefig(str(PLOT_DIR / file_name))
            plt.close()
        else:
            plt.show()


    def generate_for_process_model(process_model, process_data, parameters, size=5000, 
                                activity_dependency_ps=.25, p_var=5, seed=0, postfix=''):

        if not isinstance(p_var, Iterable):
            p_var = [p_var]

        process_map = ProcessMap.from_plg(process_model)

        np.random.seed(seed)


        # Save event log
        generator = EventLogGenerator(process_map)
        event_log = generator.generate(size=size,
                                        activity_dependency_p=activity_dependency_ps,
                                        probability_variance_max=p_var,
                                        seed=seed)
        
        if any(s in ['Machine_Steam', 'Machine_LPG', 'Machine_Diesel', 'Machine_Gasoline', 'Machine_Kerosene'] for s in list(process_data.columns)):
            attach1 = process_data[['Activity', 'Machine_Elec',
                            'kW', 'duration', 'Machine_Steam', 'Machine_LPG', 'Machine_Diesel', 'Machine_Gasoline', 'Machine_Kerosene']].drop_duplicates()

            event_log = pd.merge(event_log, attach1, how='left', on = 'Activity')
            
            ## From 최적화 코드
            np.random.seed(0)
            attach2 = pd.DataFrame({
                'Machine_Elec': np.sort(process_data['Machine_Elec'].unique()),
                'Heat' : [np.random.randint(5, 10) for i in range(0, 17)],
                'Waste' : [np.random.randint(5, 10) for i in range(0, 17)],
                'Worker' : [np.random.randint(1, 5) for i in range(0, 17)],
                'carbon_reduction_device' : [np.random.randint(200, 300) for i in range(0, 17)]
                })
            ##
                   
            event_log = pd.merge(event_log, attach2, how='left', on = 'Machine_Elec')
        else:
            attach1 = process_data[['Activity', 'Machine_Elec',
                'kW', 'duration']].drop_duplicates()
            event_log = pd.merge(event_log, attach1, how='left', on = 'Activity')

            
            
        for machine in parameters:
            for atts in machine['attributes']:
                event_log.loc[event_log.Machine_Elec == machine['Machine_Elec'], atts] =  machine['attributes'][atts]
                
        generator.plot_likelihood_graph(f'graph_{process_model}{postfix}.pdf', figsize=(20, 50))
        
        return event_log
        