#!/usr/bin/env python
#  -*- coding:utf-8 -*-

import json
from django.http import HttpResponse
import os
import socket
import time

INFO_ADDR = ('127.0.0.1', 33333)

VIDEO_ADDR = '~/www/edgeController'

def client_on(session, video, device, ip, videostarttime):
    bitrate = '0'
    chunkno = '0'
    data_json = {'session': session, 'ip': ip, 'video': video, 'device': device, 'videostarttime': videostarttime}
    jdata = json.dumps(data_json)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(120)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.connect(INFO_ADDR)
    s.send(jdata)
    try:
        bitrate, chunkno = s.recv(1024).split(' ')
    except socket.timeout as e:
        print '%s timeout' % ip
    s.close()
    return bitrate, chunkno

def client_off(session, ip):
    data_json = {'session': session, 'ip': ip}
    jdata = json.dumps(data_json)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.connect(INFO_ADDR)
    s.send(jdata)
    s.close()


def send_client_message(session, video, buf, ip):
    bitrate = '0'
    chunkno = '0'
    data_json = {'session': session, 'ip': ip, 'video': video, 'buffer': buf}
    jdata = json.dumps(data_json)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(120)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.connect(INFO_ADDR)
    s.send(jdata)
    try:
        bitrate, chunkno = s.recv(1024).split(' ')
    except socket.timeout as e:
        print '%s timeout' % ip
    s.close()
    return bitrate, chunkno

def handle_post(request):
    video_list = {}
    chunkno_list = {}
    bitrate_list = {}
    print_f = open("request_print.txt", "a")
    if request.POST:

        req = json.loads(request.body)
        session = req.get('session')#request mpd message,1 or 0 (or -1)
        ip = req.get('ip')

        if int(session) == 1:
            device = req.get('device')
            video = req.get('video')
            video_list[ip] = video
            videostart = req.get('videostarttime')
            print >> print_f, '%s %s start request chunk' % (time.time(), ip)
            print_f.flush()
            bitrate_list[ip], chunkno_list[ip] = client_on(session, video, device, ip, videostart)

            if bitrate_list[ip] != '0' and chunkno_list[ip] != '0':
                if not os.path.exists('%s/%s_%s_%s.m4s' % (VIDEO_ADDR, video_list[ip], chunkno_list[ip], bitrate_list[ip])):
                    os.system('cp /home/cjm/www/edgeController/coding_video/%s/%s_%s/segment_%s.m4s %s/%s_%s_%s.m4s' \
                              % (video_list[ip], video_list[ip], bitrate_list[ip], chunkno_list[ip], VIDEO_ADDR, \
                                 video_list[ip], chunkno_list[ip], bitrate_list[ip]))
                print >> print_f, '%s download chunk %s_%s_%s to client %s' % (time.time(), video_list[ip], chunkno_list[ip], bitrate_list[ip], ip)
                print_f.flush()

                with open('%s/%s_%s_%s.m4s' % (VIDEO_ADDR, video_list[ip], chunkno_list[ip], bitrate_list[ip]), 'r') as f:
                    res = f.read()

                print '%s return %s %s %s %s' % \
                      (time.time(), video_list[ip], video, str(chunkno_list[ip]), str(bitrate_list[ip]))
                return HttpResponse(res)
            else:
                return HttpResponse('video done')

        elif int(session) == 0:
            video = req.get('video')
            buf = req.get('buffer')
            video_list[ip] = video
            print >> print_f, '%s %s request chunk' % (time.time(), ip)
            print_f.flush()

            bitrate_list[ip], chunkno_list[ip] = send_client_message(session, video_list[ip], buf, ip)

            if bitrate_list[ip] != '0' and chunkno_list[ip] != '0':
                if not os.path.exists('%s/%s_%s_%s.m4s' % (VIDEO_ADDR, video_list[ip], chunkno_list[ip], bitrate_list[ip])):
                    os.system('cp /home/cjm/www/edgeController/coding_video/%s/%s_%s/segment_%s.m4s %s/%s_%s_%s.m4s' \
                              % (video_list[ip], video_list[ip], bitrate_list[ip], chunkno_list[ip], VIDEO_ADDR, \
                             video_list[ip], chunkno_list[ip], bitrate_list[ip]))
                print >> print_f, 'download chunk %s_%s_%s to client %s' % (video_list[ip], chunkno_list[ip], bitrate_list[ip], ip)
                print_f.flush()

                with open('%s/%s_%s_%s.m4s' % (VIDEO_ADDR, video_list[ip], chunkno_list[ip], bitrate_list[ip]), 'r') as f:
                    res = f.read()

                    print '%s return %s %s %s %s' % \
                          (time.time(), ip, video_list[ip], str(chunkno_list[ip]), str(bitrate_list[ip]))
                    return HttpResponse(res)
            else:
                return HttpResponse('video done')

        else:
            print >> print_f, '%s client close' % ip
            print_f.flush()
            client_off(session, ip)
            res = 'close'
            return HttpResponse(res)

