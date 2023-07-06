from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dropout, Activation, Flatten, Convolution1D, Dropout, Reshape
from keras.layers import Dense 
from keras.models import Sequential

import time

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):

        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)

        start = datetime.now()

        self.flow_training()

        end = datetime.now()
        print("Training time: ", (end-start))

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):

        timestamp = datetime.now()
        timestamp = timestamp.timestamp()

        file0 = open("PredictFlowStatsfile.csv","w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        body = ev.msg.body
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        for stat in sorted([flow for flow in body if (flow.priority == 1) ], key=lambda flow:
            (flow.match['eth_type'],flow.match['ipv4_src'],flow.match['ipv4_dst'],flow.match['ip_proto'])):
        
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            
            if stat.match['ip_proto'] == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']
                
            elif stat.match['ip_proto'] == 6:
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']

            elif stat.match['ip_proto'] == 17:
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']

            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)
          
            try:
                packet_count_per_second = stat.packet_count/stat.duration_sec
                packet_count_per_nsecond = stat.packet_count/stat.duration_nsec
            except:
                packet_count_per_second = 0
                packet_count_per_nsecond = 0
                
            try:
                byte_count_per_second = stat.byte_count/stat.duration_sec
                byte_count_per_nsecond = stat.byte_count/stat.duration_nsec
            except:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0
                
            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src,ip_dst, tp_dst,
                        stat.match['ip_proto'],icmp_code,icmp_type,
                        stat.duration_sec, stat.duration_nsec,
                        stat.idle_timeout, stat.hard_timeout,
                        stat.flags, stat.packet_count,stat.byte_count,
                        packet_count_per_second,packet_count_per_nsecond,
                        byte_count_per_second,byte_count_per_nsecond))
            
        file0.close()

    def flow_training(self):

        self.logger.info("Flow Training ...")

        flow_dataset = pd.read_csv('FlowStatsfile.csv')
        
        useless_columns = ['flow_id','ip_src', 'ip_dst']
        flow_dataset.drop(labels=useless_columns, axis='columns', inplace=True)
	
        X_flow = flow_dataset.iloc[:, :-1].values
        X_flow = X_flow.astype('float64')

        y_flow = flow_dataset.iloc[:, -1].values
        
        #features details
        features = flow_dataset.columns
        print('After dropping some columns: \n\t there are {} columns and {} rows'.format(len(flow_dataset.columns), len(flow_dataset)))
        
        print(flow_dataset["label"].value_counts())
        
        flow_dataset = flow_dataset.dropna().reset_index()
        flow_dataset.drop_duplicates(keep='first', inplace = True)
        print("Read {} rows.".format(len(flow_dataset)))
        
        # label encoding
        labelencoder = LabelEncoder()
        flow_dataset['label'] = labelencoder.fit_transform(flow_dataset['label'])

        print(flow_dataset['label'].value_counts())

        data_np = flow_dataset.to_numpy(dtype="float32")
        data_np = data_np[~np.isinf(data_np).any(axis=1)]        

        X = data_np[:, 0:18]
        enc = OneHotEncoder()
        Y = enc.fit_transform(data_np[:,19:]).toarray()
        
        
        print(flow_dataset.shape)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=2, shuffle=True)
        
        _features = X.shape[1]
        n_classes = Y.shape[1]

	        
        model = Sequential()

        model.add(Dense(8, input_dim=_features, activation='relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(4, activation='relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(8, activation='relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(n_classes, kernel_initializer='normal', activation='softmax'))
        #model.add(Dense(1,activation='softmax'))
        model.summary() 

        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='BinaryCrossentropy', optimizer=opt, metrics=['accuracy'])
        
        self.flow_model = model.fit(X_train, Y_train, batch_size=32, epochs=10) 
        

        # Measure inference time
        start_time = time.time()
        y_flow_pred = model.predict(X_test)
        #y_flow_pred = self.flow_model.predict(X_test)
        end_time = time.time()
        
        print("Inference time: {:.2f} seconds".format(end_time - start_time))
        
        pred = np.argmax(y_flow_pred,axis=1)
        y_test = Y_test.argmax(axis=1)
        
        self.logger.info("------------------------------------------------------------------------------")
        
        confMat = confusion_matrix(y_test, pred)
        self.logger.info(confMat)
        

        self.logger.info("------------------------------------------------------------------------------")
      
        acc = accuracy_score(y_test, pred)

        self.logger.info("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')

            #predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            #predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            #predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            useless_columns = ['flow_id','ip_src', 'ip_dst']
            predict_flow_dataset.drop(labels=useless_columns, axis='columns', inplace=True)

            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = X_predict_flow.astype('float64')
            
            y_flow_pred = self.flow_model.predict(X_predict_flow)

            legitimate_trafic = 0
            ddos_trafic = 0

            for i in y_flow_pred:
                if i == 0:
                    legitimate_trafic = legitimate_trafic + 1
                else:
                    ddos_trafic = ddos_trafic + 1
                    victim = int(predict_flow_dataset.iloc[i, 5])%20
                    
                    
                    

            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_trafic/len(y_flow_pred)*100) > 80:
                self.logger.info("legitimate trafic ...")
            else:
                self.logger.info("DDoS traffic detected!")
                self.logger.info("Victim is host: h{}".format(victim))

            self.logger.info("------------------------------------------------------------------------------")
            
            file0 = open("PredictFlowStatsfile.csv","w")
            
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            file0.close()

        except:
            pass
