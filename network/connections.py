# -*- coding: utf-8 -*-
"""
@author: juschu

Defining own connection pattern
"""


##############################
#### imports and settings ####
##############################
# import standard libraries
import math
import time

from ANNarchy import CSR

printDebug = False # print progress during creation

# minimal value for weight
MIN_CONNECTION_VALUE = 0.001


################
#### Output ####
################
class DebugOutput:
    '''
    print status of creating projection:
    current timestep and estimated elapsed time, number and percentage of created connections
    '''

    def __init__(self, connection, pre, post):
        '''
        init output class

        params: connection -- string of projection name
                pre, post  -- populations that are connected
        '''

        self.startTime = time.time()
        self.prename = pre.name
        self.postname = post.name
        self.connection = connection

        print("Create Connection {0} -> {1} with pattern {2}".format(self.prename, self.postname,
                                                                     connection))

    def Debugprint(self, value, maxvalue, connectioncreated):
        '''
        print status of creating projection

        params: value             -- current step of creating projection
                maxvalue          -- number of steps of creating projection
                connectioncreated -- number of created connections
        '''

        if printDebug:
            endtime = time.time() - self.startTime
            timestr = "%3d:%02d" % (endtime / 60, endtime % 60)

            progress = value*100.0/(maxvalue)

            if progress > 0:
                estimatedtime = endtime / progress * 100
                estimatedtimestr = "%3d:%02d" % (estimatedtime / 60, estimatedtime % 60)
            else:
                estimatedtimestr = "--:--"

            print("{0} | {1} -> {2} | {3:.4f}% | {4: <7} | {5: <7} | {6: >15}".format(
                self.connection, self.prename, self.postname,
                value*100.0/(maxvalue), timestr, estimatedtimestr, connectioncreated))


############################
#### Connection pattern ####
############################
def all2all_exp2d(pre, post, mv, sigma):
    '''
    connecting two 2-dimensional maps (normally these maps are equal)
    with gaussian field depending on distance

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            sigma     -- width of receptive field (in deg)

    return: synapses  -- CSR-object with connections
    '''

    dout = DebugOutput("all2all_exp2d", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1])
    postDimLength = (post.geometry[0], post.geometry[1])

    # Ratio of size between maps
    ratio1 = (preDimLength[0]-1)/float(postDimLength[0]-1)
    ratio2 = (preDimLength[1]-1)/float(postDimLength[1]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    # TODO: Limitation

    numOfConnectionsCreated = 0

    for post1 in range(postDimLength[0]):
        dout.Debugprint(post1, postDimLength[0], numOfConnectionsCreated)

        for post2 in range(postDimLength[1]):

            values = []
            pre_ranks = []

            for pre1 in range(preDimLength[0]):
                for pre2 in range(preDimLength[1]):

                    # distance between 2 neurons
                    dist1 = (ratio1*post1-pre1)**2
                    dist2 = (ratio2*post2-pre2)**2

                    val = mv * m_exp(-((dist1+dist2)/sigma/sigma))

                    # connect
                    numOfConnectionsCreated += 1
                    pre_rank = pre.rank_from_coordinates((pre1, pre2))
                    pre_ranks.append(pre_rank)
                    values.append(val)

            post_rank = post.rank_from_coordinates((post1, post2))
            synapses.add(post_rank, pre_ranks, values, [0])

    dout.Debugprint(1, 1, numOfConnectionsCreated)

    return synapses

def all2all_exp4d(pre, post, mv, sigma):
    '''
    connecting two 4-dimensional maps (normally these maps are equal)
    with gaussian field depending on distance

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            sigma     -- width of receptive field (in deg)

    return: synapses  -- CSR-object with connections
    '''

    dout = DebugOutput("all2all_exp4d", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1], pre.geometry[2], pre.geometry[3])
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    # Ratio of size between maps
    ratio1 = (preDimLength[0]-1)/float(postDimLength[0]-1)
    ratio2 = (preDimLength[1]-1)/float(postDimLength[1]-1)
    ratio3 = (preDimLength[2]-1)/float(postDimLength[2]-1)
    ratio4 = (preDimLength[3]-1)/float(postDimLength[3]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))
    
    numOfConnectionsCreated = 0

    for post1 in range(postDimLength[0]):
        dout.Debugprint(post1, postDimLength[0], numOfConnectionsCreated)

        for post2 in range(postDimLength[1]):
            for post3 in range(postDimLength[2]):
                for post4 in range(postDimLength[3]):

                    pre_ranks = []
                    values = []

                    # for speedup
                    rks_app = pre_ranks.append
                    vals_app = values.append

                    # limit iteration according to max_dist
                    min1 = max(0, int(math.ceil(ratio1*post1-max_dist)))
                    max1 = min(preDimLength[0]-1, int(math.floor(ratio1*post1+max_dist)))
                    min2 = max(0, int(math.ceil(ratio2*post2-max_dist)))
                    max2 = min(preDimLength[1]-1, int(math.floor(ratio2*post2+max_dist)))
                    min3 = max(0, int(math.ceil(ratio3*post3-max_dist)))
                    max3 = min(preDimLength[2]-1, int(math.floor(ratio3*post3+max_dist)))
                    min4 = max(0, int(math.ceil(ratio4*post4-max_dist)))
                    max4 = min(preDimLength[3]-1, int(math.floor(ratio4*post4+max_dist)))

                    for pre1 in range(min1, max1+1): #range(preDimLength[0]):
                        for pre2 in range(min2, max2+1): #range(preDimLength[1]):
                            for pre3 in range(min3, max3+1): #range(preDimLength[2]):
                                for pre4 in range(min4, max4+1): #range(preDimLength[3]):

                                    # distance between 2 neurons
                                    dist1 = (ratio1*post1-pre1)**2
                                    dist2 = (ratio2*post2-pre2)**2
                                    dist3 = (ratio3*post3-pre3)**2
                                    dist4 = (ratio4*post4-pre4)**2

                                    val = mv * m_exp(-((dist1+dist2+dist3+dist4)/sigma/sigma))
                                    if val > MIN_CONNECTION_VALUE:
                                        # connect
                                        numOfConnectionsCreated += 1
                                        pre_rank = pre.rank_from_coordinates((pre1, pre2, pre3, pre4))
                                        rks_app(pre_rank)
                                        vals_app(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.Debugprint(1, 1, numOfConnectionsCreated)

    return synapses


def gaussian3dTo4d(pre, post, mv, sigma):
    '''
    connecting two maps with gaussian field depending on distance of only first two dimensions
    pre map is 3d, post map is 4d

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            sigma     -- width of receptive field (in deg)

    return: synapses  -- CSR-object with connections
    '''

    dout = DebugOutput("gaussian3dTo4d", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1], pre.geometry[2])
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    # Ratio of size between maps
    ratio1 = (preDimLength[0]-1)/float(postDimLength[0]-1)
    ratio2 = (preDimLength[1]-1)/float(postDimLength[1]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numOfConnectionsCreated = 0

    for post1 in range(postDimLength[0]):
        dout.Debugprint(post1, postDimLength[0], numOfConnectionsCreated)

        for post2 in range(postDimLength[1]):
            for post3 in range(postDimLength[2]):
                for post4 in range(postDimLength[3]):

                    pre_ranks = []
                    values = []

                    # limit iteration according to max_dist
                    min1 = max(0, int(math.ceil(ratio1*post1-max_dist)))
                    max1 = min(preDimLength[0]-1, int(math.floor(ratio1*post1+max_dist)))
                    min2 = max(0, int(math.ceil(ratio2*post2-max_dist)))
                    max2 = min(preDimLength[1]-1, int(math.floor(ratio2*post2+max_dist)))

                    for pre1 in range(min1, max1+1):
                        for pre2 in range(min2, max2+1):

                            # distance between 2 neurons
                            dist1 = (ratio1*post1-pre1)**2
                            dist2 = (ratio2*post2-pre2)**2

                            val = mv * m_exp(- ((dist1+dist2)/sigma/sigma))
                            if val > MIN_CONNECTION_VALUE:
                                # connect
                                for pre3 in range(preDimLength[2]):
                                    numOfConnectionsCreated += 1
                                    pre_rank = pre.rank_from_coordinates((pre1, pre2, pre3))
                                    pre_ranks.append(pre_rank)
                                    values.append(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.Debugprint(1, 1, numOfConnectionsCreated)

    return synapses

def gaussian4dTo3d(pre, post, mv, sigma):
    '''
    connecting two maps with gaussian field depending on distance of only first two dimensions
    pre map is 4d, post map is 3d

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            sigma     -- width of receptive field (in deg)

    return: synapses  -- CSR-object with connections
    '''

    dout = DebugOutput("gaussian4dTo3d", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1], pre.geometry[2], pre.geometry[3])
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2])

    # Ratio of size between maps
    ratio1 = (preDimLength[0]-1)/float(postDimLength[0]-1)
    ratio2 = (preDimLength[1]-1)/float(postDimLength[1]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numOfConnectionsCreated = 0

    for post1 in range(postDimLength[0]):
        dout.Debugprint(post1, postDimLength[0], numOfConnectionsCreated)

        for post2 in range(postDimLength[1]):
            for post3 in range(postDimLength[2]):

                pre_ranks = []
                values = []

                # limit iteration according to max_dist
                min1 = max(0, int(math.ceil(ratio1*post1-max_dist)))
                max1 = min(preDimLength[0]-1, int(math.floor(ratio1*post1+max_dist)))
                min2 = max(0, int(math.ceil(ratio2*post2-max_dist)))
                max2 = min(preDimLength[1]-1, int(math.floor(ratio2*post2+max_dist)))

                for pre1 in range(min1, max1+1):
                    for pre2 in range(min2, max2+1):

                        # distance between 2 neurons
                        dist1 = (ratio1*post1-pre1)**2
                        dist2 = (ratio2*post2-pre2)**2

                        val = mv * m_exp(- ((dist1+dist2)/sigma/sigma))
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            for pre3 in range(preDimLength[2]):
                                for pre4 in range(preDimLength[3]):
                                    numOfConnectionsCreated += 1
                                    pre_rank = pre.rank_from_coordinates((pre1, pre2, pre3, pre4))
                                    pre_ranks.append(pre_rank)
                                    values.append(val)

                post_rank = post.rank_from_coordinates((post1, post2, post3))
                synapses.add(post_rank, pre_ranks, values, [0])

    dout.Debugprint(1, 1, numOfConnectionsCreated)

    return synapses

def gaussian2dTo4d_h(pre, post, mv, sigma):
    '''
    connect two maps with a gaussian receptive field 2d to 4d
    independent of last two dimension of 4d map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            sigma     -- width of receptive field (in deg)

    return: synapses  -- CSR-object with connections
    '''

    dout = DebugOutput("gaussian2dTo4d_h", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1])
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    # Ratio of size between maps
    ratio1 = (preDimLength[0]-1)/float(postDimLength[0]-1)
    ratio2 = (preDimLength[1]-1)/float(postDimLength[1]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numOfConnectionsCreated = 0

    #w_post
    for post1 in range(postDimLength[0]):
        dout.Debugprint(post1, postDimLength[0], numOfConnectionsCreated)

        for post2 in range(postDimLength[1]):
            for post3  in range(postDimLength[2]):
                for post4 in range(postDimLength[3]):

                    pre_ranks = []
                    values = []

                    # limit iteration according to max_dist
                    min1 = max(0, int(math.ceil(ratio1*post1-max_dist)))
                    max1 = min(preDimLength[0]-1, int(math.floor(ratio1*post1+max_dist)))
                    min2 = max(0, int(math.ceil(ratio2*post2-max_dist)))
                    max2 = min(preDimLength[1]-1, int(math.floor(ratio2*post2+max_dist)))

                    for pre1 in range(min1, max1+1): #range(preDimLength[0]):
                        for pre2 in range(min2, max2+1): #range(preDimLength[1]):

                            dist1 = (ratio1*post1-pre1)**2
                            dist2 = (ratio2*post2-pre2)**2

                            val = mv * m_exp(-((dist1+dist2)/sigma/sigma))
                            if val > MIN_CONNECTION_VALUE:
                                # connect
                                numOfConnectionsCreated += 1
                                pre_rank = pre.rank_from_coordinates((pre1, pre2))
                                pre_ranks.append(pre_rank)
                                values.append(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.Debugprint(1, 1, numOfConnectionsCreated)

    return synapses

def gaussian2dTo4d_v(pre, post, mv, sigma):
    '''
    connect two maps with a gaussian receptive field 2d to 4d
    independent of first two dimension of 4d map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            sigma     -- width of receptive field (in deg)

    return: synapses  -- CSR-object with connections
    '''

    dout = DebugOutput("gaussian2dTo4d_v", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1])
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    # Ratio of size between maps
    ratio1 = (preDimLength[0]-1)/float(postDimLength[2]-1)
    ratio2 = (preDimLength[1]-1)/float(postDimLength[3]-1)

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numOfConnectionsCreated = 0

    for post1 in range(postDimLength[0]):
        dout.Debugprint(post1, postDimLength[0], numOfConnectionsCreated)

        for post2  in range(postDimLength[1]):
            for post3  in range(postDimLength[2]):
                for post4  in range(postDimLength[3]):

                    # limit iteration according to max_dist
                    min1 = max(0, int(math.ceil(ratio1*post3-max_dist)))
                    max1 = min(preDimLength[0]-1, int(math.floor(ratio1*post3+max_dist)))
                    min2 = max(0, int(math.ceil(ratio2*post4-max_dist)))
                    max2 = min(preDimLength[1]-1, int(math.floor(ratio2*post4+max_dist)))

                    values = []
                    pre_ranks = []

                    for pre1 in range(min1, max1+1): #range(preDimLength[0]):
                        for pre2 in range(min2, max2+1): #range(preDimLength[1]):

                            dist1 = (ratio1*post3-pre1)**2
                            dist2 = (ratio2*post4-pre2)**2

                            val = mv * m_exp(-((dist1+dist2)/sigma/sigma))
                            if val > MIN_CONNECTION_VALUE:
                                # connect
                                numOfConnectionsCreated += 1
                                pre_rank = pre.rank_from_coordinates((pre1, pre2))
                                pre_ranks.append(pre_rank)
                                values.append(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.Debugprint(1, 1, numOfConnectionsCreated)

    return synapses

def gaussian2dTo4d_diag(pre, post, mv, sigma):
    '''
    connect two maps with a gaussian receptive field 2d to 4d diagonally

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            sigma     -- width of receptive field (in deg)

    return: synapses  -- CSR-object with connections
    '''

    dout = DebugOutput("gaussian2dTo4d_diag", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1])
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    offset1 = (preDimLength[0]-1)/2.0
    offset2 = (preDimLength[1]-1)/2.0

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numConnectionCreated = 0

    for post1 in range(postDimLength[0]):
        dout.Debugprint(post1, postDimLength[0], numConnectionCreated)

        for post2 in range(postDimLength[1]):
            for post3 in range(postDimLength[2]):
                for post4 in range(postDimLength[3]):

                    values = []
                    pre_ranks = []

                    # limit iteration according to max_dist
                    min1 = max(0, int(math.ceil(post1+post3-offset1-max_dist)))
                    max1 = min(preDimLength[0]-1, int(math.floor(post1+post3-offset1+max_dist)))
                    min2 = max(0, int(math.ceil(post2+post4-offset2-max_dist)))
                    max2 = min(preDimLength[1]-1, int(math.floor(post2+post4-offset2+max_dist)))

                    for pre1 in range(min1, max1+1): #range(preDimLength[0]):
                        for pre2 in range(min2, max2+1): #range(preDimLength[1]):

                            dist1 = (post1 + post3 - pre1 - offset1)**2
                            dist2 = (post2 + post4 - pre2 - offset2)**2

                            val = mv * m_exp(-((dist1+dist2)/sigma/sigma))
                            if val > MIN_CONNECTION_VALUE:
                                # connect
                                numConnectionCreated += 1

                                pre_rank = pre.rank_from_coordinates((pre1, pre2))
                                pre_ranks.append(pre_rank)
                                values.append(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.Debugprint(1, 1, numConnectionCreated)

    return synapses

def gaussian4dTo2d_diag(pre, post, mv, sigma):
    '''
    connect two maps with a gaussian receptive field 4d to 2d diagonally

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            sigma     -- width of receptive field (in deg)

    return: synapses  -- CSR-object with connections
    '''

    dout = DebugOutput("gaussian4dTo2d_diag", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1], pre.geometry[2], pre.geometry[3])
    postDimLength = (post.geometry[0], post.geometry[1])

    offset1 = (preDimLength[0]-1)/2.0
    offset2 = (preDimLength[1]-1)/2.0

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numOfConnectionsCreated = 0

    for post1 in range(postDimLength[0]):
        dout.Debugprint(post1, postDimLength[0], numOfConnectionsCreated)

        for post2  in range(postDimLength[1]):

            values = []
            pre_ranks = []

            for pre1 in range(preDimLength[0]):
                for pre2 in range(preDimLength[1]):

                    # limit iteration according to max_dist
                    min1 = max(0, int(math.ceil(post1+offset1-pre1-max_dist)))
                    max1 = min(preDimLength[2]-1, int(math.floor(post1+offset1-pre1+max_dist)))
                    min2 = max(0, int(math.ceil(post2+offset2-pre2-max_dist)))
                    max2 = min(preDimLength[3]-1, int(math.floor(post2+offset2-pre2+max_dist)))

                    for pre3 in range(min1, max1+1): #range(preDimLength[2]):
                        for pre4 in range(min2, max2+1): #range(preDimLength[3]):

                            dist1 = (post1 - (pre1+pre3) + offset1)**2
                            dist2 = (post2 - (pre2+pre4) + offset2)**2

                            val = mv * m_exp(-((dist1+dist2)/sigma/sigma))
                            if val > MIN_CONNECTION_VALUE:
                                # connect
                                numOfConnectionsCreated += 1
                                pre_rank = pre.rank_from_coordinates((pre1, pre2, pre3, pre4))
                                pre_ranks.append(pre_rank)
                                values.append(val)

            post_rank = post.rank_from_coordinates((post1, post2))
            synapses.add(post_rank, pre_ranks, values, [0])

    dout.Debugprint(1, 1, numOfConnectionsCreated)

    return synapses

def gaussian4d_diagTo4d_v(pre, post, mv, sigma):
    '''
    connect two maps with a gaussian receptive field 4d to 4d diagonally
    independent of first two dimension of post map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            sigma     -- width of receptive field (in deg)

    return: synapses  -- CSR-object with connections
    '''

    dout = DebugOutput("gaussian4d_diagTo4d_v", pre, post)

    preDimLength = (pre.geometry[0], pre.geometry[1], pre.geometry[2], pre.geometry[3])
    postDimLength = (post.geometry[0], post.geometry[1], post.geometry[2], post.geometry[3])

    offset1 = (preDimLength[0]-1)/2.0
    offset2 = (preDimLength[1]-1)/2.0

    synapses = CSR()

    # for speedup
    m_exp = math.exp
    max_dist = sigma * math.sqrt(-math.log(MIN_CONNECTION_VALUE/mv))

    numConnectionCreated = 0

    for post1 in range(postDimLength[0]):
        for post2 in range(postDimLength[1]):
            dout.Debugprint(post1 * postDimLength[1]  + post2, postDimLength[0]*postDimLength[1], numConnectionCreated)

            for post3 in range(postDimLength[2]):
                for post4 in range(postDimLength[3]):

                    values = []
                    pre_ranks = []

                    for pre1 in range(preDimLength[0]):
                        for pre2 in range(preDimLength[1]):

                            # limit iteration according to max_dist
                            min1 = max(0, int(math.ceil(post3+offset1-pre1-max_dist)))
                            max1 = min(preDimLength[2]-1, int(math.floor(post3+offset1-pre1+max_dist)))
                            min2 = max(0, int(math.ceil(post4+offset2-pre2-max_dist)))
                            max2 = min(preDimLength[3]-1, int(math.floor(post4+offset2-pre2+max_dist)))

                            for pre3 in range(min1, max1+1): #range(preDimLength[2]):
                                for pre4 in range(min2, max2+1): #range(preDimLength[3]):

                                    if (pre != post) or ((post3 != pre3) and (post4 != pre4)):

                                        dist1 = (post3 - (pre1+pre3) + offset1)**2
                                        dist2 = (post4 - (pre2+pre4) + offset2)**2

                                        val = mv * m_exp(-((dist1+dist2)/sigma/sigma))
                                        if val > MIN_CONNECTION_VALUE:
                                            # connect
                                            numConnectionCreated += 1
                                            pre_rank = pre.rank_from_coordinates((pre1, pre2, pre3, pre4))
                                            pre_ranks.append(pre_rank)
                                            values.append(val)

                    post_rank = post.rank_from_coordinates((post1, post2, post3, post4))
                    synapses.add(post_rank, pre_ranks, values, [0])

    dout.Debugprint(1, 1, numConnectionCreated)

    return synapses
