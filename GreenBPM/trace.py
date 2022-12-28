# Copyright 2022. Jonghyeon Ko
#
# All scripts in this folder are distributed as a free software. 
# We extended a few functions for fundamental concepts of process mining introduced in the complementary repository for BINet [1].
# 
# 1. [Nolle, T., Seeliger, A., Mühlhäuser, M.: BINet: Multivariate Business Process Anomaly Detection Using Deep Learning, 2018](https://doi.org/10.1007/978-3-319-98648-7_16)
# ==============================================================================

import uuid


class Case(object):
    def __init__(self, id=None, events=None, **kwargs):
        self.id = id
        if events is None:
            self.events = []
        else:
            self.events = events
        self.attributes = dict(kwargs)

    def add_event(self, event):
        self.events.append(event)


class Event(object):
    def __init__(self, name, timestamp=None, timestamp_end=None, **kwargs):
        self.name = name
        self.timestamp = timestamp
        self.timestamp_end = timestamp_end
        self.attributes = dict(kwargs)
        

class Trace(object):
    """Base class for anomaly implementations."""

    def __init__(self):
        self.graph = None
        self.activities = None
        self.attributes = None
        self.name = self.__class__.__name__[:-7]

    def __str__(self):
        return self.name


    @staticmethod
    def apply_to_case(self, case):
        pass


    def apply_to_path(self, path):
        return self.path_to_case(path)


    def path_to_case(self, p, label=None):
        g = self.graph

        case = Case(label=label)
        for i in range(0, len(p)):
            event = Event(name=g.nodes[p[i]]['value'])
            att = g.nodes[p[i]]['name']
            value = g.nodes[p[i]]['value']
            event.attributes[att] = value
            case.add_event(event)
        return case
    
    
    def path_to_trace(self, p, label=None):
        g = self.graph
        trace = [ uuid.uuid1(g.nodes[p[i]]['value']) for i in range(0, len(p))]  
        return trace


class Apply_Trace(Trace):

    def __init__(self):
        super(Apply_Trace, self).__init__()
        self.name = 'Normal'

    def apply_to_case(self, case):
        case.attributes['label'] = 'normal'
        return case
    
    def apply_to_trace(self, case):
        return case
