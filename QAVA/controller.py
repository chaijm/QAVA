#!/usr/bin/env python
#  -*- coding:utf-8 -*-
 
import urllib2
import json
import threading
import socket
import time
from arguments import get_args
from A3C import central_agent, A3C
import multiprocessing as mp
from numpy import *

# need to change to the appropriate IP address
HTTPSERVER_URL = 'http://IP/'
CHUNK_ADDR = 'http://IP/opt/video/vod/coding_video'
VIDEO_ADDR = '~/www/edgeController'

# port to listen for client state information
CLIENT_INFO_ADDRESS = ('127.0.0.1', 33333)

BITRATE = [334, 396, 522, 595, 791, 1000, 1200, 1500, 2100, 2500]

# need to change to the duration of all videos
VIDEO_DURATION = {}
# need to change to the names of predicted video quality files
PREDICT_QUALITY_FILE = {}

REBUFFER_PENALTY = 0.5
DELAY_PENALTY = 0.03
SMOOTH_PENALTY = 1
CHUNK_SKIP_PENALTY = 0.3
HIGH_PENALTY = 0.8
LOW_PENALTY = 1
THROUGHPUT_LIST = [0.0] * 8 #past 8 throughput samples
CLIENT_NUM = 20
PH_NUM = 10
CLIENT_CONN = {}
conn_mutex = threading.Lock()
reward_mutex = threading.Lock()
throughput_mutex = threading.Lock()
MAX_BANDWIDTH = 4 #Mbps
A3CDict = {}
DELAY_TOLERATE = 0


def InitialNet(video_name, state_space_len, action_space_len, net_params_queue, exp_queue, lock):
    A3CDict[video_name] = A3C(video_name, state_space_len, action_space_len,
                              net_params_queue, exp_queue, lock)
    while True:
        time.sleep(120)

class Client():
    def __init__(self, v, d, ip):
        self.device = d
        self.video = v
        self.ip = ip
        self.chunkno = 0
        self.past_chunkno = 0
        self.buffer = 0.0
        self.rebuffer = 0.0
        self.delay = 0.0
        self.chunk_skip = 0
        self.quality = 0.0
        self.past_quality = 0.0
        self.info_time = time.time() #client request time
        self.qoe = 0.0
        self.newrequest = False

    def _set_client_message(self, v, buf):
        self.video = v
        self.buffer = buf
        self.info_time = time.time()
        self.newrequest = True

class Video():
    def __init__(self, v, start_time):
        self.start_time = start_time
        self.video_name = v
        self.chunkno = self._compute_request_chunkno()
        self.past_chunkno = 0
        self.pp_chunkno = 0
        self.past_ph_clients = 0
        self.past_tv_clients = 0
        self.bitrate = 0
        self.past_bitrate = 0
        self.past_chunk_ph_quality = 0.0
        self.past_chunk_tv_quality = 0.0
        self.pp_chunk_ph_quality = 0.0
        self.pp_chunk_tv_quality = 0.0
        self.past_chunk_download_time = 0.0
        self.past_chunk_download_speed = 0.0
        self.chunk_download = False
        self.clients_mutex = threading.Lock()
        self.clients = {}
        self.next_clients_mutex = threading.Lock()
        self.next_clients = {}
        self.pastc_qoe_mutex = threading.Lock()
        self.past_clients = {}
        self.past_clients_qoe = {}
        self.past_clients_qoe_rawdata = {}
        self.reward = 0
        self.delay_list = [0.0] * 8
        self.delay_mutex = threading.Lock()

    def _compute_request_chunkno(self):
        return int((time.time() - self.start_time) / 2) + 1

    def _compute_request_wait_time(self):
        next_chunk_aviliable_time = self.chunkno * 2 + self.start_time
        return max(0, next_chunk_aviliable_time - time.time())

    def _get_chunk_quality(self, bitrate):
        f = open('%s/chunkQuality/%s.mp4_ph_vmaf.log' % (VIDEO_ADDR, self.video_name), 'r')
        lines = f.readlines()
        line = lines[self.chunkno - 1].strip('\n')
        chunk_qualities = (line.split(','))[6:16]
        self.pp_chunk_ph_quality = self.past_chunk_ph_quality
        self.past_chunk_ph_quality = float(chunk_qualities[BITRATE.index(bitrate)])
        f.close()

        f = open('%s/chunkQuality/%s.mp4_tv_vmaf.log' % (VIDEO_ADDR, self.video_name), 'r')
        lines = f.readlines()
        line = lines[self.chunkno - 1].strip('\n')
        chunk_qualities = (line.split(','))[6:16]
        self.pp_chunk_tv_quality = self.past_chunk_tv_quality
        self.past_chunk_tv_quality = float(chunk_qualities[BITRATE.index(bitrate)])
        f.close()


class Controller():
    def __init__(self):
        self.totalclients = {}
        self.totalvideos = {}
        self.totalvideos_lock = threading.Lock()
        self.state_space_len = 48
        self.action_space_len = len(BITRATE)

        self.args = get_args()
        self.online_agents_num = mp.Value('i', 0)
        self.online_id_vector = mp.Array('i', [0] * self.args.total_agents)
        self.lock = mp.Lock()
        # inter-process communication queues
        self.net_params_queues = []
        self.exp_queues = []
        for i in range(self.args.total_agents):
            self.net_params_queues.append(mp.Queue(maxsize=0))
            self.exp_queues.append(mp.Queue(maxsize=0))

        p = threading.Thread(target=central_agent,
                       args=(self.net_params_queues, self.exp_queues,
                             self.state_space_len, self.action_space_len, self.online_id_vector,
                             self.online_agents_num, self.lock))
        p.start()

        video_params_id = 0
        for video_name in VIDEO_DURATION.keys():
            t = threading.Thread(target=InitialNet, args=(video_name, self.state_space_len, self.action_space_len,
                                                          self.net_params_queues[video_params_id],
                                                          self.exp_queues[video_params_id], self.lock))
            t.start()
            video_params_id += 1

        listen_client_info_t = threading.Thread(target=self._listen_client_info, args=())
        listen_client_info_t.start()
        compute_fairness_t = threading.Thread(target=self._compute_fairness, args=())
        compute_fairness_t.start()
        compute_throughput_t = threading.Thread(target=self._compute_throughput, args=())
        compute_throughput_t.start()

    def _listen_client_info(self):
        client_info_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_info_s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        client_info_s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        client_info_s.bind(CLIENT_INFO_ADDRESS)
        # max session number
        client_info_s.listen(2000)
        while True:
            conn, addr = client_info_s.accept()
            info = json.loads(conn.recv(1024))
            session = info.get('session')
            ip = info.get('ip')
            print '%s make a connection' % ip

            # clients start to request videos
            if int(session) == 1:
                video_name = info.get('video')
                device = int(info.get('device'))
                # for the sake of simplicity, let client directly upload the video start time from the MPD file
                video_start_time = float(info.get('videostarttime'))
                print '%s %s start request %s' % (time.time(), ip, video_name)
                c = Client(video_name, device, ip)
                self.totalclients[ip] = c
                if video_name in self.totalvideos.keys():
                    if self.totalvideos[video_name].start_time != video_start_time:
                        self.totalvideos[video_name].start_time = video_start_time
                else:
                    # delete redundant obsolete chunks
                    os.system('rm -f %s/%s.mpd' % (VIDEO_ADDR, video_name))
                    os.system('rm -f %s/%s_*.m4s' % (VIDEO_ADDR, video_name))
                    v = Video(video_name, video_start_time)
                    self.totalvideos_lock.acquire()
                    self.totalvideos[video_name] = v
                    self.totalvideos_lock.release()

                    #open a new video thread
                    t = threading.Thread(target=self._open_video_thread, args=(video_name,))
                    t.setName(video_name)
                    t.start()

                # To reduce the startup time, edge returns the latest chunk that has been downloaded to clients
                if self.totalvideos[video_name].past_chunkno != 0 and self.totalvideos[video_name].past_bitrate != 0 and \
                        self.totalvideos[video_name].chunk_download == False:
                    conn.send(str(self.totalvideos[video_name].past_bitrate)+' '+str(self.totalvideos[video_name].past_chunkno))
                    c.past_chunkno = c.chunkno = self.totalvideos[video_name].past_chunkno

                    if c.device == 1:
                        c.past_quality = c.quality = self.totalvideos[video_name].past_chunk_ph_quality
                    else:
                        c.past_quality = c.quality = self.totalvideos[video_name].past_chunk_tv_quality
                    conn.close()

                    print 'send past chunk %s %s to %s' % \
                          (str(c.past_chunkno), str(self.totalvideos[video_name].past_bitrate), str(c.ip))
                else:
                    conn_mutex.acquire()
                    CLIENT_CONN[ip] = conn
                    conn_mutex.release()
                    c.newrequest = True
                    c.chunkno = self.totalvideos[video_name].chunkno
                    self.totalvideos[video_name].clients_mutex.acquire()
                    self.totalvideos[video_name].clients[ip] = c

                    self.totalvideos[video_name].clients_mutex.release()

                    print '%s wait newest chunk %s' % (str(c.ip), str(c.chunkno))


            elif int(session) == 0:
                video_name = info.get('video')
                buf = float(info.get('buffer'))
                self.totalclients[ip]._set_client_message(video_name, buf)

                # update video info
                video = self.totalvideos[video_name]
                self.totalclients[ip].chunkno = video._compute_request_chunkno()

                conn_mutex.acquire()
                CLIENT_CONN[ip] = conn
                conn_mutex.release()

                if self.totalclients[ip].chunkno == self.totalclients[ip].past_chunkno:
                    print '%s need wait next chunk availiable' % ip
                    video.next_clients_mutex.acquire()
                    video.next_clients[ip] = self.totalclients[ip]
                    video.next_clients_mutex.release()
                elif self.totalclients[ip].chunkno > self.totalclients[ip].past_chunkno \
                        and self.totalclients[ip].chunkno >= video.chunkno:
                    print '%s wait this chunk download' % ip
                    self.totalclients[ip].chunkno = video.chunkno
                    video.clients_mutex.acquire()
                    video.clients[ip] = self.totalclients[ip]
                    video.clients_mutex.release()
                elif self.totalclients[ip].chunkno > self.totalclients[ip].past_chunkno \
                        and self.totalclients[ip].chunkno < video.chunkno \
                        and video.past_bitrate != 0 and video.past_chunkno != 0:
                    conn_mutex.acquire()
                    if ip in CLIENT_CONN.keys():
                        (CLIENT_CONN[ip]).send(str(video.past_bitrate) + ' ' + str(video.past_chunkno))
                        (CLIENT_CONN[ip]).close()
                        del CLIENT_CONN[ip]
                    conn_mutex.release()
                    self.totalclients[ip].past_chunkno = self.totalclients[ip].chunkno
                    if video.chunk_download == True:
                        self.totalclients[ip].past_quality = self.totalclients[ip].quality
                        if self.totalclients[ip].device == 1:
                            self.totalclients[ip].quality = video.pp_chunk_ph_quality
                        else:
                            self.totalclients[ip].quality = video.pp_chunk_tv_quality
                    else:
                        self.totalclients[ip].past_quality = self.totalclients[ip].quality
                        if self.totalclients[ip].device == 1:
                            self.totalclients[ip].quality = video.past_chunk_ph_quality
                        else:
                            self.totalclients[ip].quality = video.past_chunk_tv_quality
                    self.totalclients[ip].chunkno = video.past_chunkno
                else:
                    video.clients_mutex.acquire()
                    client = self.totalclients[ip]
                    client.past_chunkno = client.chunkno = video.chunkno
                    video.clients[ip] = self.totalclients[ip]
                    video.clients_mutex.release()
            else:
                conn.close()
                if ip in self.totalclients.keys():
                    del self.totalclients[ip]
                print '%s stop request' % ip

    def _open_video_thread(self, video_name):
        total_reward = 0
        total_reward_num = 0
        flag = -1
        video = self.totalvideos[video_name]
        a3c = A3CDict[video.video_name]

        if not (os.path.exists('%s/%s.mpd' % (VIDEO_ADDR, video.video_name))):
            try:
                url = HTTPSERVER_URL + video.video_name + '.mpd'
                req = urllib2.Request(url)
                res_data = urllib2.urlopen(req, timeout=60)
                res = res_data.read()
            except:
                print 'mpd download error'

            os.system('cp %s/coding_video/%s/%s.mpd %s/%s.mpd' \
                      % (VIDEO_ADDR, video.video_name, video.video_name, VIDEO_ADDR, video.video_name))

        while time.time() - video.start_time < VIDEO_DURATION[video.video_name]:
            wait_start_time = time.time()
            # wait util client requests chunks
            request_wait_time = video._compute_request_wait_time()
            while len(video.clients) == 0 \
                and (time.time() - wait_start_time) < request_wait_time \
                and (time.time() - video.start_time) < VIDEO_DURATION[video.video_name]:
                time.sleep(0.0001)
            if time.time() - video.start_time >= VIDEO_DURATION[video.video_name]:
                print '%s timeout' % video_name
                break

            if video.past_chunkno != 0:
                # update client qoe
                print '%s compute reward' % video_name
                total_delay = 0
                total_delay_num = 0
                for client in video.past_clients.values():
                    client.chunkno = video.chunkno
                    client.chunk_skip = max((video.chunkno - (client.past_chunkno + 1)), 0)
                    while (not client.newrequest) and (time.time() - video.start_time) < VIDEO_DURATION[video.video_name]:
                        time.sleep(0.0001)
                    if time.time() - video.start_time >= VIDEO_DURATION[video.video_name]:
                        print '%s timeout' % video_name
                        break
                    t = time.time()
                    newest_chunkno = int((t - video.start_time) / 2) + 1
                    newest_chunk_time = (newest_chunkno - 1) * 2 + video.start_time
                    client.delay = max(0, client.buffer - (t - client.info_time)) + \
                                   (max(0, newest_chunkno - (client.past_chunkno + 1))) * 2 + t - newest_chunk_time

                    total_delay += client.delay
                    total_delay_num += 1

                    client.qoe = client.quality - SMOOTH_PENALTY * abs(client.quality - client.past_quality) \
                                 - REBUFFER_PENALTY * client.rebuffer - CHUNK_SKIP_PENALTY * client.chunk_skip \
                                 - DELAY_PENALTY * client.delay
                    video.pastc_qoe_mutex.acquire()
                    video.past_clients_qoe[client.ip] = client.qoe
                    video.past_clients_qoe_rawdata[client.ip] = [client.quality, client.past_quality, \
                                                                 client.rebuffer, client.delay, client.chunk_skip]
                    video.pastc_qoe_mutex.release()
                    f = open('%s_QoE.txt' % client.ip, 'a')
                    f.write('%s %s %s %s %s %s %s %s %s %s %s\n' % \
                            (time.time(), client.video, client.past_chunkno, bitrate, client.quality, \
                             client.past_quality, client.buffer, client.rebuffer, client.delay, \
                             client.chunk_skip, client.qoe))
                    f.flush()
                    f.close()
                if time.time() - video.start_time >= VIDEO_DURATION[video.video_name]:
                    print '%s timeout' % video_name
                    break
                else:
                    video.delay_mutex.acquire()
                    video.delay_list.remove(video.delay_list[0])
                    if total_delay_num > 0:
                        video.delay_list.append(float(total_delay) / total_delay_num / 8) # /8s
                    else:
                        video.delay_list.append(0.0)  # /8s
                    video.delay_mutex.release()
                    # reward
                    video.reward = self._compute_video_reward(video)
                    a3c.save_reward(video.reward)
                    total_reward = total_reward + video.reward
                    total_reward_num = total_reward_num + 1
                    print '%s a3c save reward %s' % (video_name, str(video.reward))

                    # train
                    a3c.train(False)

                    flag = 1
            if time.time() - video.start_time >= VIDEO_DURATION[video.video_name]:
                print '%s timeout' % video_name
                break


            state = self._video_state(video)

            video.past_ph_clients = 0
            video.past_tv_clients = 0

            bitrate = BITRATE[a3c.action(state)]
            video._get_chunk_quality(bitrate)

            flag = 0

            video.past_bitrate = video.bitrate
            video.bitrate = bitrate
            rawdata_f = open('%s_rawdata.txt' % video.video_name, 'a')
            rawdata_f.write(str(bitrate) + ' ')
            rawdata_f.flush()
            rawdata_f.close()

            video.chunk_download = False

            # download this bitrate's chunk
            self._request_new_chunk(video, bitrate, video.video_name, video.chunkno)

            video.chunk_download = True
            wait_start_time = time.time()
            # wait util
            request_wait_time = video._compute_request_wait_time()

            while ((len(video.clients) != 0) or (time.time() - wait_start_time) < request_wait_time)\
                    and time.time() - video.start_time < VIDEO_DURATION[video.video_name]:
                if len(video.clients) != 0:
                    video.clients_mutex.acquire()
                    for client in video.clients.values():
                        client.newrequest = False
                        if client.device == 1:
                            video.past_ph_clients = video.past_ph_clients + 1
                        else:
                            video.past_tv_clients = video.past_tv_clients + 1

                        # compute client metrics
						if client.past_chunkno == 0:
                            if client.device == 1:
                                client.past_quality = client.quality = video.past_chunk_ph_quality
                            else:
                                client.past_quality = client.quality = video.past_chunk_tv_quality
                        else:
                            client.past_quality = client.quality
                            if client.device == 1:
                                client.quality = video.past_chunk_ph_quality
                            else:
                                client.quality = video.past_chunk_tv_quality
								
                        if client.past_chunkno == 0:
                            client.buffer = 2
                            client.rebuffer = 0

                        else:
                            downloadtime = time.time() - client.info_time
                            if downloadtime > client.buffer:
                                client.rebuffer = downloadtime - client.buffer
                                client.buffer = 2
                            else:
                                client.buffer = client.buffer - downloadtime + 2
                                client.rebuffer = 0


                        client.chunkno = video.chunkno
                        client.past_chunkno = client.chunkno
                        video.past_clients[client.ip] = client

                        conn_mutex.acquire()
                        if client.ip in CLIENT_CONN.keys():
                            (CLIENT_CONN[client.ip]).send(str(bitrate) + ' ' + str(client.chunkno))
                            (CLIENT_CONN[client.ip]).close()
                            del CLIENT_CONN[client.ip]
                        conn_mutex.release()
                        del video.clients[client.ip]

                    video.clients_mutex.release()
                else:
                    time.sleep(0.0001)
            if time.time() - video.start_time >= VIDEO_DURATION[video.video_name]:
                if len(video.clients) > 0:
                    video.clients_mutex.acquire()
                    for client in video.clients.values():
                        client.newrequest = False
                        if client.device == 1:
                            video.past_ph_clients = video.past_ph_clients + 1
                        else:
                            video.past_tv_clients = video.past_tv_clients + 1

                        # compute client qoe
						if client.past_chunkno == 0:
                            if client.device == 1:
                                client.past_quality = client.quality = video.past_chunk_ph_quality
                            else:
                                client.past_quality = client.quality = video.past_chunk_tv_quality
                        else:
                            client.past_quality = client.quality
                            if client.device == 1:
                                client.quality = video.past_chunk_ph_quality
                            else:
                                client.quality = video.past_chunk_tv_quality
								
                        client.chunk_skip = 0
                        if client.past_chunkno == 0:
                            client.buffer = 2
                            client.rebuffer = 0

                        else:
                            downloadtime = time.time() - client.info_time
                            if downloadtime > client.buffer:
                                client.rebuffer = downloadtime - client.buffer
                                client.buffer = 2
                            else:
                                client.buffer = client.buffer - downloadtime + 2
                                client.rebuffer = 0

                        client.chunkno = video.chunkno
                        client.past_chunkno = client.chunkno
                        video.past_clients[client.ip] = client

                        conn_mutex.acquire()
                        if client.ip in CLIENT_CONN.keys():
                            (CLIENT_CONN[client.ip]).send(str(bitrate) + ' ' + str(client.chunkno))
                            (CLIENT_CONN[client.ip]).close()
                            del CLIENT_CONN[client.ip]
                        conn_mutex.release()
                        del video.clients[client.ip]
                    video.clients_mutex.release()
                break

            if video.past_chunkno == 0 and video.pp_chunkno == 0:
                video.pp_chunkno = video.past_chunkno = video.chunkno
            else:
                video.pp_chunkno = video.past_chunkno
                video.past_chunkno = video.chunkno
            chunkno = video._compute_request_chunkno()
            if chunkno - (video.past_chunkno + 1) <= DELAY_TOLERATE:
                video.chunkno = video.past_chunkno + 1
            else:
                video.chunkno = chunkno
            if len(video.next_clients) != 0:
                video.clients_mutex.acquire()
                video.next_clients_mutex.acquire()
                for client in video.next_clients.values():
                    client.chunkno = video.chunkno
                video.clients.update(video.next_clients)
                video.next_clients = {}
                video.next_clients_mutex.release()
                video.clients_mutex.release()

        print 'video %s is done' % video_name
        time.sleep(1)
        if len(video.clients) != 0:
            for client in video.clients.values():
                conn_mutex.acquire()
                if client.ip in CLIENT_CONN.keys():
                    (CLIENT_CONN[client.ip]).send(str(0) + ' ' + str(0))
                    (CLIENT_CONN[client.ip]).close()
                    del CLIENT_CONN[client.ip]
                conn_mutex.release()
        if len(video.next_clients) != 0:
            for client in video.next_clients.values():
                conn_mutex.acquire()
                if client.ip in CLIENT_CONN.keys():
                    (CLIENT_CONN[client.ip]).send(str(0) + ' ' + str(0))
                    (CLIENT_CONN[client.ip]).close()
                    del CLIENT_CONN[client.ip]
                conn_mutex.release()

        if flag == 0:
            # update client qoe
            for client in video.past_clients.values():
                client.chunk_skip = 0

                client.delay = client.buffer

                client.qoe = client.quality - SMOOTH_PENALTY * abs(client.quality - client.past_quality) \
                             - REBUFFER_PENALTY * client.rebuffer - CHUNK_SKIP_PENALTY * client.chunk_skip \
                             - DELAY_PENALTY * client.delay
                video.pastc_qoe_mutex.acquire()
                video.past_clients_qoe[client.ip] = client.qoe
                video.past_clients_qoe_rawdata[client.ip] = [client.quality, client.past_quality, \
                                                             client.rebuffer, client.delay, client.chunk_skip]
                video.pastc_qoe_mutex.release()
                f = open('%s_QoE.txt' % client.ip, 'a')
                f.write('%s %s %s %s %s %s %s %s %s %s %s\n' % \
                        (time.time(), client.video, client.past_chunkno, bitrate, client.quality, \
                         client.past_quality, client.buffer, client.rebuffer, client.delay, \
                         client.chunk_skip, client.qoe))
                f.flush()
                f.close()
            # reward
            video.reward = self._compute_video_reward(video)
            a3c.save_reward(video.reward)
            total_reward = total_reward + video.reward
            total_reward_num = total_reward_num + 1
            print '%s a3c save reward' % video_name

            # train
            a3c.train(True)

        ttreward_f = open('%s_totalreward.txt' % video.video_name, 'a')
        if total_reward_num > 0:
            ttreward_f.write(str(time.time()) + ' ' + str(float(total_reward)/total_reward_num) + '\n')
        else:
            ttreward_f.write(str(0) + '\n')
        ttreward_f.flush()
        ttreward_f.close()
        if video.video_name in self.totalvideos.keys():
            self.totalvideos_lock.acquire()
            del self.totalvideos[video_name]
            self.totalvideos_lock.release()
        print 'video thread %s close' % video_name

    def _get_net_data(self):
        nc = '/proc/net/dev'
        fd = open(nc, "r")
        netcardstatus = False
        for line in fd.readlines():
            # need to change to the appropriate netcard
            if "enp129s0f0" in line:
                netcardstatus = True
                field = line.split()
                recv = field[1]
        if not netcardstatus:
            fd.close()
            print 'please setup your netcard'
            sys.exit()
        fd.close()
        return float(recv)

    def _compute_throughput(self):
        recv = self._get_net_data()
        time_interval = 2

        while True:
            time.sleep(time_interval)
            new_recv = self._get_net_data()
            recvdata = float(new_recv - recv) * 8 / 1024 / 1024 / time_interval #Mbps
            throughput_mutex.acquire()
            THROUGHPUT_LIST.remove(THROUGHPUT_LIST[0])
            THROUGHPUT_LIST.append(float(recvdata)/MAX_BANDWIDTH)
            throughput_mutex.release()
            recv = new_recv


    def _client_qoe_difference(self, video):
        qoe_high = 0
        qoe_low = 0
        video.pastc_qoe_mutex.acquire()
        vid_qoe = []
        for qoe_value in video.past_clients_qoe_rawdata.values():
            vid_qoe.append(qoe_value[0])
        for i in range(len(vid_qoe)):
            if vid_qoe[i] < -1:
                vid_qoe[i] = -1
        if len(vid_qoe) == 0:
            vid_qoe_mean = 0
        else:
            vid_qoe_mean = np.mean(vid_qoe)

        v = []
        for ip in video.past_clients_qoe_rawdata.keys():
            id = int(ip.split('.')[-1])
            tmp_v = [id] + video.past_clients_qoe_rawdata[ip]
            v.append(tmp_v)

        video.pastc_qoe_mutex.release()
        whole_qoe = {}
        whole_qoe_mean = []
        w = {}
        all_video_name = []
        for client in self.totalclients.values():
            if client.video in self.totalvideos.keys():
                video_name = client.video
                all_video_name.append(video_name)
                quality = client.quality
                past_quality = client.past_quality
                quality_switch = abs(quality - past_quality)
                gap_time = time.time() - client.info_time
                if gap_time > client.buffer:
                    rebuffer = gap_time - client.buffer
                    tmp_video = self.totalvideos[video_name]
                    now_chunkno = tmp_video._compute_request_chunkno()
                    if now_chunkno > tmp_video.chunkno and now_chunkno - (tmp_video.chunkno + 1) <= DELAY_TOLERATE:
                        now_chunkno = tmp_video.chunkno + 1
                    if now_chunkno > tmp_video.chunkno:
                        chunk_skip = max(0, now_chunkno - (tmp_video.chunkno + 1))
                    else:
                        chunk_skip = 0
                else:
                    rebuffer = 0
                    chunk_skip = 0
                t = time.time()
                tmp_video = self.totalvideos[video_name]
                newest_chunkno = int((t - tmp_video.start_time) / 2) + 1
                newest_chunk_time = (newest_chunkno - 1) * 2 + tmp_video.start_time
                delay = max(0, client.buffer - (t - client.info_time)) + \
                        (max(0, newest_chunkno - (client.past_chunkno + 1))) * 2 + t - newest_chunk_time

                qoe = quality - 0*SMOOTH_PENALTY * quality_switch \
                      - 0*REBUFFER_PENALTY * rebuffer - 0*CHUNK_SKIP_PENALTY * chunk_skip - 0*DELAY_PENALTY * delay

                if qoe < -1:
                    qoe = -1
                id = int(client.ip.split('.')[-1])
                if video_name in whole_qoe:
                    whole_qoe[video_name].append(qoe)
                    w[video_name].append([id, quality, past_quality, rebuffer, delay, chunk_skip])
                else:
                    whole_qoe[video_name] = [qoe]
                    w[video_name] = [[id, quality, past_quality, rebuffer, delay, chunk_skip]]
        whole_qoe[video.video_name] = vid_qoe
        w[video.video_name] = v
        for video_name in whole_qoe.keys():
            if len(whole_qoe[video_name]) == 0:
                whole_qoe_mean.append(0)
            else:
                whole_qoe_mean.append(np.mean(whole_qoe[video_name]))


        vid_len = len(vid_qoe)
        data_len = len(whole_qoe_mean)
        if data_len <= 1 or vid_len < 1:
            return v, w, qoe_high, qoe_low
        

        high_num = 0
        low_num = 0
        for q in whole_qoe_mean:
            if vid_qoe_mean > q:
                qoe_high += vid_qoe_mean - q
                high_num += 1
            else:
                qoe_low += q - vid_qoe_mean
                low_num += 1

        if high_num + low_num > 1:
            qoe_high = qoe_high / (high_num + low_num -1)
            qoe_low = qoe_low / (high_num + low_num - 1)

        return v, w, qoe_high, qoe_low


    def _video_state(self, video):
        rawdata_f = open('%s_rawdata.txt' % video.video_name, 'a')
        rawdata = str(time.time()) + ' ' + str(video.chunkno) + ' '

        state = []

        throughput_mutex.acquire()
        state = state + THROUGHPUT_LIST
        rawdata = rawdata + ' '.join(str(i * MAX_BANDWIDTH) for i in THROUGHPUT_LIST) + ' '
        throughput_mutex.release()

        totalbitrate = 0
        for v in self.totalvideos.values():
            if v.chunk_download == False:
                totalbitrate = totalbitrate + int(v.bitrate)
        totalbitrate = float(totalbitrate) / 1024

        state = state + [float(totalbitrate) / MAX_BANDWIDTH]
        rawdata = rawdata + str(totalbitrate) + ' '

        v, w, qoe_high, qoe_low = self._client_qoe_difference(video)
        state = state + [qoe_high, qoe_low]
        rawdata = rawdata + str(len(v)) + ' '
        rawdata = rawdata + ' '.join(str(j) for i in v for j in i) + ' '
        rawdata = rawdata + str(len(w)) + ' '
        for vn in w.keys():
            rawdata = rawdata + str(vn) + ' ' + str(len(w[vn])) + ' '
            rawdata = rawdata + ' '.join(str(j) for i in w[vn] for j in i) + ' '

        state = state + [float(video.past_ph_clients) / max(1, PH_NUM), float(video.past_tv_clients) / (CLIENT_NUM-PH_NUM)]
        rawdata = rawdata + str(video.past_ph_clients) + ' ' + str(video.past_tv_clients) + ' '

        f = open('%s/predictQuality/%s_ph_preValue.txt' % \
                 (VIDEO_ADDR, PREDICT_QUALITY_FILE[video.video_name]), 'r')
        lines = f.readlines()
        line = lines[max(video.chunkno - 2, 0)].strip('\n')
        next_chunk_ph_quality = (line.split(','))
        for i in range(len(next_chunk_ph_quality)):
            next_chunk_ph_quality[i] = float(next_chunk_ph_quality[i])
        state = state + next_chunk_ph_quality
        rawdata = rawdata + ' '.join(str(i) for i in next_chunk_ph_quality) + ' '

        f = open('%s/predictQuality/%s_tv_preValue.txt' % \
                 (VIDEO_ADDR, PREDICT_QUALITY_FILE[video.video_name]), 'r')
        lines = f.readlines()
        line = lines[max(video.chunkno - 2, 0)].strip('\n')
        next_chunk_tv_quality = (line.split(','))
        for i in range(len(next_chunk_tv_quality)):
            next_chunk_tv_quality[i] = float(next_chunk_tv_quality[i])
        state = state + next_chunk_tv_quality
        rawdata = rawdata + ' '.join(str(i) for i in next_chunk_tv_quality) + ' '

        state = state + [video.past_chunk_ph_quality]
        rawdata = rawdata + str(video.past_chunk_ph_quality) + ' '

        state = state + [video.past_chunk_tv_quality]
        rawdata = rawdata + str(video.past_chunk_tv_quality) + ' '

        state = state + [video.past_chunk_download_speed / MAX_BANDWIDTH]
        rawdata = rawdata + str(video.past_chunk_download_speed) + ' '

        state = state + [video.past_chunk_download_time / 10]
        rawdata = rawdata + str(video.past_chunk_download_time) + ' '

        last_chunk_ship = max(video.chunkno - (video.past_chunkno + 1), 0)
        if last_chunk_ship == 0:
            state = state + [0, 0]
        elif last_chunk_ship == 1 or last_chunk_ship == 2:
            state = state + [0, 1]
        else:
            state = state + [1, 1]
        rawdata = rawdata + str(last_chunk_ship) + ' '

        chunk_stay_time = time.time() - ((video.chunkno - 1) * 2 + video.start_time)
        if chunk_stay_time < 0:
            chunk_stay_time = 0
        state = state + [chunk_stay_time / 2]
        rawdata = rawdata + str(chunk_stay_time) + ' '

        video.delay_mutex.acquire()
        tmp_delay_list = [0.0] * 8
        for i in range(len(video.delay_list)):
            if video.delay_list[i] > 1:
                tmp_delay_list[i] = 1.0
            else:
                tmp_delay_list[i] = video.delay_list[i]
        state = state + tmp_delay_list
        rawdata = rawdata + ' '.join(str(i * 8) for i in video.delay_list) + ' '
        video.delay_mutex.release()

        state = np.array(state)

        rawdata_f.write(rawdata)
        rawdata_f.flush()
        rawdata_f.close()
        return state

    def _compute_video_reward(self, video):
        record_reward_f = open('%s_reward.txt' % video.video_name, 'a')
        v, w, qoe_high, qoe_low = self._client_qoe_difference(video)

        vid_qoe = []
        rebuf = []
        for q in v:
            quality = float(q[1])
            past_quality = float(q[2])
            rebuffer = float(q[3])
            delay = float(q[4])
            chunk_skip = float(q[5])
            qoe = quality - SMOOTH_PENALTY * abs(quality - past_quality) \
                  - REBUFFER_PENALTY * rebuffer - CHUNK_SKIP_PENALTY * chunk_skip -DELAY_PENALTY * delay
            if qoe < -1:
                qoe = -1
            vid_qoe.append(qoe)
            rebuf.append(rebuffer)

        reward = np.mean(vid_qoe) - HIGH_PENALTY * qoe_high - LOW_PENALTY * qoe_low
        record_reward_f.write(str(time.time()) + ' ' + str(np.mean(vid_qoe)) + ' ' + \
                              str(qoe_high) + ' ' + str(qoe_low) + ' ' + str(np.mean(rebuf)) + \
                              ' ' + str(reward) + '\n')
        record_reward_f.flush()
        record_reward_f.close()

        rawdata_f = open('%s_rawdata.txt' % video.video_name, 'a')
        rawdata = str(len(v)) + ' '
        rawdata = rawdata + ' '.join(str(j) for i in v for j in i) + ' '
        rawdata = rawdata + str(len(w)) + ' '
        for vn in w.keys():
            rawdata = rawdata + str(vn) + ' ' + str(len(w[vn])) + ' '
            rawdata = rawdata + ' '.join(str(j) for i in w[vn] for j in i) + ' '

        rawdata = rawdata + '\n'
        rawdata_f.write(rawdata)
        rawdata_f.flush()
        rawdata_f.close()

        return reward

    def _request_new_chunk(self, video, bitrate, video_name, chunkno):
        stime = time.time()
        # send request to server
        try:
            url = '%s/%s/%s_%s/segment_%s.m4s' % (CHUNK_ADDR, video_name, video_name, bitrate, chunkno)
            req = urllib2.Request(url)
            res_data = urllib2.urlopen(req, timeout=60)
            res = res_data.read()

            video.past_chunk_download_time = time.time() - stime
            video.past_chunk_download_speed = len(res) * 8.0 / video.past_chunk_download_time / 1000000
        except:
            print 'chunk download error'
        os.system('cp %s/coding_video/%s/%s_%s/segment_%s.m4s %s/%s_%s_%s.m4s' \
                  % (VIDEO_ADDR, video_name, video_name, bitrate, chunkno, VIDEO_ADDR, video_name, chunkno, bitrate))
        print "%s %s %s already downloaded" % (video_name, chunkno, bitrate)

    def _compute_fairness(self):
        while True:
            if len(self.totalclients) != 0:
                qoe = []
                printqoe = {}
                for client in self.totalclients.values():
                    if client.video in self.totalvideos.keys():
                        quality = client.quality
                        past_quality = client.past_quality
                        quality_switch = abs(quality - past_quality)
                        gap_time = time.time() - client.info_time
                        if gap_time > client.buffer:
                            rebuffer = gap_time - client.buffer
                            video = self.totalvideos[client.video]
                            now_chunkno = video._compute_request_chunkno()
                            if now_chunkno > video.chunkno and now_chunkno - (video.chunkno + 1) <= DELAY_TOLERATE:
                                now_chunkno = video.chunkno + 1
                            if now_chunkno > video.chunkno:
                                chunk_skip = max(0, now_chunkno - (video.chunkno + 1))
                            else:
                                chunk_skip = 0
                        else:
                            rebuffer = 0
                            chunk_skip = 0
                        t = time.time()
                        video = self.totalvideos[client.video]
                        newest_chunkno = int((t - video.start_time) / 2) + 1
                        newest_chunk_time = (newest_chunkno - 1) * 2 + video.start_time
                        delay = max(0, client.buffer - (t - client.info_time)) + \
                                (max(0, newest_chunkno - (client.past_chunkno + 1))) * 2 + t - newest_chunk_time
                        client_qoe = quality - SMOOTH_PENALTY * quality_switch \
                              - REBUFFER_PENALTY * rebuffer - CHUNK_SKIP_PENALTY * chunk_skip - DELAY_PENALTY * delay
                        qoe.append(client_qoe)
                        printqoe[client.ip] = client_qoe

                if len(qoe) != 0:
                    qoe_mean = np.mean(qoe)
                    qoe_std = np.std(qoe)
                    min_qoe = min(qoe)
                    f = open('allQoE.txt', 'a')
                    f.write('%s\n' % str(printqoe))
                    f.write('%s %s %s %s\n' % \
                            (time.time(), qoe_mean, qoe_std, min_qoe))
                    f.flush()
                    f.close()
            time.sleep(1)

if __name__ == "__main__":
    os.system('rm -f %s/*.m4s' % VIDEO_ADDR)
    os.system('rm -f %s/*.mpd' % VIDEO_ADDR)

    edge_controller = Controller()
    print 'open controller'
