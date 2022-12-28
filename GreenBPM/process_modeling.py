# Copyright 2022. Jonghyeon Ko
#
# All scripts in this folder are distributed as a free software. 
# We extended a few functions for fundamental concepts of process mining introduced in the complementary repository for BINet [1].
# 
# 1. [Nolle, T., Seeliger, A., Mühlhäuser, M.: BINet: Multivariate Business Process Anomaly Detection Using Deep Learning, 2018](https://doi.org/10.1007/978-3-319-98648-7_16)
# ==============================================================================


import os

import pandas as pd
import networkx as nx
import numpy as np
import untangle
from matplotlib import pyplot as plt
from datetime import datetime

from GreenBPM.dir import PLOT_DIR
from GreenBPM.dir import IMG_DIR
from GreenBPM.dir import TEMP_DIR
from GreenBPM.dir import PROCESS_MODEL_DIR

import seaborn as sns
microsoft_colors = sns.color_palette(['#01b8aa', '#374649', '#fd625e', '#f2c80f', '#5f6b6d',
                                      '#8ad4eb', '#fe9666', '#a66999', '#3599b8', '#dfbfbf', '#008631', '#98BF64'])

from matplotlib import font_manager, rc
# 한글폰트작업
# window의 폰트 위치 -> C:/Windows/Fonts/NGULIM.TTF
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/NGULIM.TTF").get_name()
plt.rc('font', family=font_name)
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.image as mpimg

imgfoot = mpimg.imread(str(IMG_DIR / 'foot.png'))
imgLift = mpimg.imread(str(IMG_DIR / 'Lift.png'))
imgDryer = mpimg.imread(str(IMG_DIR /'Dryer.png'))
imgCleaner = mpimg.imread(str(IMG_DIR / 'Cleaner.png'))
imgBoiler = mpimg.imread(str(IMG_DIR / 'Boiler.png'))
imgLPG = mpimg.imread(str(IMG_DIR / 'LPG2.png'))
imgHeat = mpimg.imread(str(IMG_DIR / 'steam2.png'))
imgDiesel = mpimg.imread(str(IMG_DIR / 'Diesel2.png'))
imgKero = mpimg.imread(str(IMG_DIR / 'Kerosene2.png'))
imgOthers = mpimg.imread(str(IMG_DIR / 'others.png'))
imgLine = mpimg.imread(str(IMG_DIR / 'line.png'))

start_symbol = '▶'
end_symbol = '■' 

class ProcessMap(object):
    def __init__(self, graph=None): 
        self.graph = graph
        self.start_event = start_symbol
        self.end_event = end_symbol


    @staticmethod
    def from_plg(file_path):
        """Load a process model from a plg file (the format PLG2 uses).

        Gates will be ignored in the resulting process map.

        :param file_path: path to plg file
        :return: ProcessMap object
        """

        if not file_path.endswith('.plg'):
            file_path += '.plg'
        if not os.path.isabs(file_path):
            file_path = os.path.join(PROCESS_MODEL_DIR, file_path)

        with open(file_path, encoding='utf-8') as f:
            file_content = untangle.parse(f.read())

        start_event = int(file_content.process.elements.startEvent['id'])
        end_event = int(file_content.process.elements.endEvent['id'])

        id_activity = dict((int(task['id']), str(task['name'])) for task in file_content.process.elements.task)
        id_activity[start_event] = start_symbol
        id_activity[end_event] = end_symbol

        activities = id_activity.keys()

        gateways = [int(g['id']) for g in file_content.process.elements.gateway]
        gateway_followers = dict((id_, []) for id_ in gateways)
        followers = dict((id_, []) for id_ in activities)

        for sf in file_content.process.elements.sequenceFlow:
            source = int(sf['sourceRef'])
            target = int(sf['targetRef'])
            if source in gateways:
                gateway_followers[source].append(target)

        for sf in file_content.process.elements.sequenceFlow:
            source = int(sf['sourceRef'])
            target = int(sf['targetRef'])
            if source in activities and target in activities:
                followers[source].append(target)
            elif source in activities and target in gateways:
                followers[source] = gateway_followers.get(target)

        graph = nx.DiGraph()
        graph.add_nodes_from([id_activity.get(activity) for activity in activities])
        for source, targets in followers.items():
            for target in targets:
                graph.add_edge(id_activity.get(source), id_activity.get(target))

        return ProcessMap(graph)

    def plot_process_map(self, name=None, figsize=None):
        g = self.graph

        # Draw
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')

        # Set figure size
        if figsize is None:
            figsize = (8, 8)
        fig = plt.figure(3, figsize=figsize)

        color_map = []
        for node in g:
            if node in [start_symbol, end_symbol]:
                color_map.append(microsoft_colors[0])
            else:
                color_map.append(microsoft_colors[2])

        nx.draw(g, pos, node_color=color_map, with_labels=True,font_family=font_name, font_size=12)
        if name is not None:
            # Save to disk
            plt.tight_layout()
            fig.savefig(str(PLOT_DIR / name))
            plt.close()
        else:
            plt.show()


    def plot_process_map_GreenBPM(self, data, name=None, figsize=None, entire_period = None, seed= 1234):
        g = self.graph
        # Draw
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')

        # Set figure size / entire_period
        if figsize is None:
            figsize = (11, 11)
        fig = plt.figure(3, figsize=figsize)
        
        if entire_period is None:
            entire_period = 52
        
        color_map = []
        node_sizes = []
        
        # 1달 간 공장가동 20일 가정
        if any("Timestamp" in s for s in list(data.columns)):
            entire_period = (max(np.array([datetime.strptime(i, '%Y-%m-%d %H:%M') for i in data.Complete_Timestamp ]))- 
                            min(np.array([datetime.strptime(i, '%Y-%m-%d %H:%M') for i in data.Start_Timestamp ])) ).total_seconds()/(24*60*60)
            one_month = 20/entire_period
            print("Total period:", entire_period, "days")
        else:
            one_month = 20/entire_period
            print("Total period:", entire_period, "days")
        
        if any(s in ['duration'] for s in list(data.columns)):
            pass
        else:
            data['duration'] =  data.apply(lambda row: (datetime.strptime(row['Complete_Timestamp'], '%Y-%m-%d %H:%M') - 
                                            datetime.strptime(row['Start_Timestamp'], '%Y-%m-%d %H:%M')).total_seconds()/(60*60), axis=1 )
            
        data['kWh'] = data.kW*data.duration
        kWh_max = max(data['kWh'])
        kWh_min = min(data['kWh'])
        dict_kWh = dict(zip(data.Activity, data.kWh))

        ## From 최적화 코드: CO2 계산식 (*414 에서 *424로 수정함)
        if any(s in ['Heat', 'Waste', 'Worker'] for s in list(data.columns)):
            np.random.seed(seed)
            df_before_variables = data[['Machine_Elec', 'Heat', 'Waste', 'Worker']].drop_duplicates()
            df_before_variables['CO2(kg)'] = df_before_variables.apply(lambda row: 
                np.random.normal(data[data['Machine_Elec'] == row['Machine_Elec']]['kWh'].iloc[0] * 424, (row['Heat'] + row['Waste'])/row['Worker'])/1000000, axis=1)
        else:
            np.random.seed(seed)
            df_before_variables = data[['Machine_Elec']].drop_duplicates()
            df_before_variables['CO2(kg)'] = df_before_variables.apply(lambda row: 
                data[data['Machine_Elec'] == row['Machine_Elec']]['kWh'].iloc[0] * 424/1000000, axis=1)

        
        ## 
        
        data = pd.merge(data, df_before_variables)
        data.to_csv(str(TEMP_DIR / name) + '.csv', index= False)
        # 1달 간 공장가동 20일 가정
        df_kWhm = pd.DataFrame( { 'Activity': data.groupby('Activity')['kWh'].sum().index , 
                             'kWh/m':  np.round(data.groupby('Activity')['kWh'].sum().values*one_month, 2) })
        dict_kWhm = dict(zip( df_kWhm.Activity, df_kWhm['kWh/m']))
        
        df_CO2m = pd.DataFrame( { 'Activity': data.groupby('Activity')['CO2(kg)'].sum().index , 
                             'CO2(kg)/m':  np.round(data.groupby('Activity')['CO2(kg)'].sum().values*one_month, 2) })        
        dict_CO2m = dict(zip( df_CO2m.Activity, df_CO2m['CO2(kg)/m']))
        
        ## 전력사용량에 의한 탄소발자국 프로세스
        for node in g:
            if node in [start_symbol, end_symbol]:
                color_map.append(microsoft_colors[0])
                node_sizes.append(400)
            else:
                color_map.append(microsoft_colors[11])
                node_sizes.append( round( 500 + 1000*((dict_kWh[node] - kWh_min)/(kWh_max - kWh_min))**5 ,2))
        
        
        nx.draw(g, pos, node_color=color_map, with_labels=True, 
                font_family=font_name, font_size=12, node_size = node_sizes)

        ## 전력사용량 외 다른 온실가스 배출원
        if any(s in ['Heat', 'Waste', 'Worker'] for s in list(data.columns)):
        ### To do: 직접연소시설 정의 바뀌면 수정
            dict_LPG = dict(zip( data.Activity, data.Machine_LPG))
            dict_Diesel = dict(zip( data.Activity, data.Machine_Diesel))
            dict_Kerosene = dict(zip( data.Activity, data.Machine_Kerosene))
            dict_Steam = dict(zip( data.Activity, data.Machine_Steam))
            
            df_Diesel_m = pd.DataFrame( { 'key': data.groupby('Machine_Diesel')['CO2(kg)'].sum().index , 
                                'CO2(kg)/m':  np.round(data.groupby('Machine_Diesel')['CO2(kg)'].sum().values*one_month,2) })
            dict_Diesel_m = dict(zip( df_Diesel_m.key, df_Diesel_m['CO2(kg)/m']))
            
            df_Kerosene_m = pd.DataFrame( { 'key': data.groupby('Machine_Kerosene')['CO2(kg)'].sum().index , 
                                'CO2(kg)/m':  np.round(data.groupby('Machine_Kerosene')['CO2(kg)'].sum().values*one_month,2) })
            dict_Kerosene_m = dict(zip( df_Kerosene_m.key, df_Kerosene_m['CO2(kg)/m']))
            
            df_LPG_m = pd.DataFrame( { 'key': data.groupby('Machine_LPG')['CO2(kg)'].sum().index , 
                                'CO2(kg)/m':  np.round(data.groupby('Machine_LPG')['CO2(kg)'].sum().values*one_month,2) })
            dict_LPG_m = dict(zip( df_LPG_m.key, df_LPG_m['CO2(kg)/m']))

            df_Steam_m = pd.DataFrame( { 'key': data.groupby('Machine_Steam')['CO2(kg)'].sum().index , 
                                'CO2(kg)/m':  np.round(data.groupby('Machine_Steam')['CO2(kg)'].sum().values*one_month,2) })
            dict_Steam_m = dict(zip( df_Steam_m.key, df_Steam_m['CO2(kg)/m']))
        
            ax=plt.gca()
            fig=plt.gcf()
            trans = ax.transData.transform
            trans2 = fig.transFigure.inverted().transform
            imsize = 0.12 # this is the image size for foot
            cache_kero_bolier = list()
            cache_lpg_dryer = list()
            cache_steam_cleaner = list()
            for key in g.nodes():
                if key in [start_symbol]:
                    (x,y) = pos[key]
                    xx,yy = trans((x,y)) # figure coordinates
                    xa,ya = trans2((xx,yy)) # axes coordinates
                    a = plt.axes([xa-imsize/2.0 ,ya-imsize/2.0 + 0.05 , imsize, imsize ])
                    a.imshow(imgfoot)
                    a.set_aspect('equal')
                    a.axis('off')
                    c = plt.axes([0.83 + 0.35 ,ya-imsize/2.0 + 0.015 , 0.13, 0.13 ])
                    c.imshow(imgOthers)
                    c.set_aspect('equal')
                    c.axis('off')
                    
                elif key in [end_symbol]:
                    (x,y) = pos[key]
                    xx,yy = trans((x,y)) # figure coordinates
                    xa,ya = trans2((xx,yy)) # axes coordinates
                    a = plt.axes([xa-imsize/2.0 ,ya-imsize/2.0 - 0.05 , imsize, imsize ])
                    a.imshow(imgfoot)
                    a.set_aspect('equal')
                    a.axis('off')
                    b = plt.axes([xa ,ya - 0.10 , 0, 0 ])
                    b.set_aspect('equal')
                    b.axis('off')
                    plt.text(xa, ya, 
                            "{'Total kWh': '" + str(round(sum(data.kWh)*one_month,2)) + ", 'Total CO2': '"+ str(round( sum(data['CO2(kg)'])*one_month ,2)) + "'}")
                else:
                    (x,y) = pos[key]
                    xx,yy = trans((x,y)) # figure coordinates
                    xa,ya = trans2((xx,yy)) # axes coordinates
                    a = plt.axes([xa ,ya - 0.03  , 0.00, 0.00 ])
                    a.set_aspect('equal')
                    a.axis('off')
                    plt.text(xa, ya, 
                            "{'kWh': '" + str(round(dict_kWhm[key],2)) + ", 'CO2(kg)': '" + str(round(dict_CO2m[key] ,2)) + "'}")
                    if "Lift" in str(dict_Diesel[key]) : 
                        (x,y) = pos[key]
                        xx,yy = trans((x,y)) # figure coordinates
                        xa,ya = trans2((xx,yy)) # axes coordinates
                        line = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.07, 0.3, 0.3 ])
                        line.imshow(imgLine)
                        line.set_aspect('equal')
                        line.axis('off')
                        
                        a = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 + 0.035, 0.04, 0.04 ])
                        a.imshow(imgLift)
                        a.set_aspect('equal')
                        a.axis('off')
                        
                        b = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.01 , 0.07, 0.07 ])
                        b.imshow(imgDiesel)
                        b.set_aspect('equal')
                        b.axis('off')
                        c = plt.axes([0.83 + 0.45 ,ya-imsize/2.0 + 0.06 , 0.00, 0.00 ])
                        c.set_aspect('equal')
                        c.axis('off')
                        plt.text(0.85 + 0.4, ya-imsize/2.0 , 
                                "{'Lift for " + str(key) + ", 'CO2(kg)': '" + str(round(dict_Diesel_m[dict_Diesel[key]] ,2)) + "'}")
                    
                    elif "Boiler" in str(dict_Kerosene[key]) : 
                        (x,y) = pos[key]
                        xx,yy = trans((x,y)) # figure coordinates
                        xa,ya = trans2((xx,yy)) # axes coordinates
                        line = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.07, 0.3, 0.3 ])
                        line.imshow(imgLine)
                        line.set_aspect('equal')
                        line.axis('off')
                        a = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 + 0.035, 0.04, 0.04 ])
                        a.imshow(imgBoiler)
                        a.set_aspect('equal')
                        a.axis('off')
                        b = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.01 , 0.07, 0.07 ])
                        b.imshow(imgKero)
                        b.set_aspect('equal')
                        b.axis('off')
                        
                        cache_kero_bolier.append(key)
                        c = plt.axes([0.83 + 0.45 ,ya-imsize/2.0 + 0.06 - 0.015*(len(cache_kero_bolier)-1), 0.00, 0.00 ])
                        c.set_aspect('equal')
                        c.axis('off')
                        plt.text(0.85 + 0.5, ya-imsize/2.0, 
                                "{'Boiler for " + str(key) + ",'CO2(kg)': '" + str(round(dict_Kerosene_m[dict_Kerosene[key]] ,2)) + "'}")
                        
                    elif "Boiler" in str(dict_LPG[key]) : 
                        (x,y) = pos[key]
                        xx,yy = trans((x,y)) # figure coordinates
                        xa,ya = trans2((xx,yy)) # axes coordinates
                        line = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.07, 0.3, 0.3 ])
                        line.imshow(imgLine)
                        line.set_aspect('equal')
                        line.axis('off')
                        a = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 + 0.035, 0.04, 0.04 ])
                        a.imshow(imgBoiler)
                        a.set_aspect('equal')
                        a.axis('off')
                        b = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.01 , 0.07, 0.07 ])
                        b.imshow(imgLPG)
                        b.set_aspect('equal')
                        b.axis('off')
                        c = plt.axes([0.83 + 0.45 ,ya-imsize/2.0 + 0.06 , 0.00, 0.00 ])
                        c.set_aspect('equal')
                        c.axis('off')
                        plt.text(0.85 + 0.4, ya-imsize/2.0 , 
                                "{'Boiler for " + str(key) + ",'CO2(kg)': '" + str(round(dict_LPG_m[dict_LPG[key]] ,2)) + "'}")                
            

                    elif "Cleaner" in str(dict_Steam[key]) : 
                        (x,y) = pos[key]
                        xx,yy = trans((x,y)) # figure coordinates
                        xa,ya = trans2((xx,yy)) # axes coordinates
                        line = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.07, 0.3, 0.3 ])
                        line.imshow(imgLine)
                        line.set_aspect('equal')
                        line.axis('off')
                        a = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 + 0.035, 0.04, 0.04 ])
                        a.imshow(imgCleaner)
                        a.set_aspect('equal')
                        a.axis('off')
                        b = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.01 , 0.07, 0.07 ])
                        b.imshow(imgHeat)
                        b.set_aspect('equal')
                        b.axis('off')
                        
                        cache_steam_cleaner.append(key)
                        c = plt.axes([0.83 + 0.45 ,ya-imsize/2.0 + 0.06 - 0.015*(len(cache_steam_cleaner)-1), 0.00, 0.00 ])
                        c.set_aspect('equal')
                        c.axis('off')
                        plt.text(0.85 + 0.4, ya-imsize/2.0 , 
                                "{'Cleaner for " + str(key) + ", 'CO2(kg)': '" + str(round(dict_Steam_m[dict_Steam[key]] ,2)) + "'}")   
                        
                        
                    elif "Dryer" in str(dict_LPG[key]) : 
                        (x,y) = pos[key]
                        xx,yy = trans((x,y)) # figure coordinates
                        xa,ya = trans2((xx,yy)) # axes coordinates
                        line = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.07, 0.3, 0.3 ])
                        line.imshow(imgLine)
                        line.set_aspect('equal')
                        line.axis('off')
                        a = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 + 0.035, 0.04, 0.04 ])
                        a.imshow(imgDryer)
                        a.set_aspect('equal')
                        a.axis('off')
                        b = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.01 , 0.07, 0.07 ])
                        b.imshow(imgLPG)
                        b.set_aspect('equal')
                        b.axis('off')
                        
                        if len(cache_lpg_dryer) > 2: # To be updated
                            cache_lpg_dryer = list()
                            
                        cache_lpg_dryer.append(key)
                        c = plt.axes([0.83 + 0.45 ,ya-imsize/2.0 + 0.06 - 0.015*(len(cache_lpg_dryer)-1), 0.00, 0.00 ])
                        c.set_aspect('equal')
                        c.axis('off')
                        plt.text(0.85 + 0.4, ya-imsize/2.0 , 
                                "{'Dryer for " + str(key) + ", 'CO2(kg)': '" + str(round(dict_LPG_m[dict_LPG[key]] ,2)) + "'}")
                        
                    else :
                        (x,y) = pos[key]
                        xx,yy = trans((x,y)) # figure coordinates
                        xa,ya = trans2((xx,yy)) # axes coordinates
                        line = plt.axes([0.78 + 0.4 ,ya-imsize/2.0 - 0.07, 0.3, 0.3 ])
                        line.imshow(imgLine)
                        line.set_aspect('equal')
                        line.axis('off')         
            
            
        else: # 단순 전력 설비만 존재할 때
            ax=plt.gca()
            fig=plt.gcf()
            trans = ax.transData.transform
            trans2 = fig.transFigure.inverted().transform
            imsize = 0.12 # this is the image size for foot
            cache_kero_bolier = list()
            cache_lpg_dryer = list()
            cache_steam_cleaner = list()
            for key in g.nodes():
                if key in [start_symbol]:
                    (x,y) = pos[key]
                    xx,yy = trans((x,y)) # figure coordinates
                    xa,ya = trans2((xx,yy)) # axes coordinates
                    a = plt.axes([xa-imsize/2.0 ,ya-imsize/2.0 + 0.05 , imsize, imsize ])
                    a.imshow(imgfoot)
                    a.set_aspect('equal')
                    a.axis('off')
                elif key in [end_symbol]:
                    (x,y) = pos[key]
                    xx,yy = trans((x,y)) # figure coordinates
                    xa,ya = trans2((xx,yy)) # axes coordinates
                    a = plt.axes([xa-imsize/2.0 ,ya-imsize/2.0 - 0.05 , imsize, imsize ])
                    a.imshow(imgfoot)
                    a.set_aspect('equal')
                    a.axis('off')
                    b = plt.axes([xa ,ya - 0.10 , 0, 0 ])
                    b.set_aspect('equal')
                    b.axis('off')
                    plt.text(xa, ya, 
                            "{'Total kWh': '" + str(round(sum(data.kWh)*one_month,2)) + ", 'Total CO2': '"+ str(round( sum(data['CO2(kg)'])*one_month ,2)) + "'}")

                else:
                    (x,y) = pos[key]
                    xx,yy = trans((x,y)) # figure coordinates
                    xa,ya = trans2((xx,yy)) # axes coordinates
                    a = plt.axes([xa ,ya - 0.03  , 0.00, 0.00 ])
                    a.set_aspect('equal')
                    a.axis('off')
                    plt.text(xa, ya, 
                            "{'kWh': '" + str(round(dict_kWhm[key],2)) + ", 'CO2(kg)': '" + str(round(dict_CO2m[key] ,2)) + "'}") 
                    
                    
                    
        if name is not None:
            # Save to disk
            plt.show()
            fig.savefig(str(PLOT_DIR / name), bbox_inches='tight')
            plt.close()
        else:
            plt.show()