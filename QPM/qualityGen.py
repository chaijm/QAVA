import os
import sys
import json

# generate VMAF score

BITRATE = [334, 396, 522, 595, 791, 1000, 1200, 1500, 2100, 2500]

WIDTH, HEIGHT = 1920, 1080
FPS = 24

def computer_score(file_name):
    s = []
    with open(file_name) as json_file:
        data = json.load(json_file)
        data = data['frames']
        chunk_score = 0
        for i in range(len(data)):
            score = float(data[i]['VMAF_score']) / 100.0
            chunk_score += score
            if (int(data[i]['frameNum'])+1) % (FPS * 2) == 0 or i == len(data)-1:
                if len(data) % (FPS * 2) != 0 and (i == len(data) - 1):
                    chunk_score = chunk_score / (len(data) % (FPS * 2))
                else:
                    chunk_score = chunk_score / (FPS * 2)
                s.append(chunk_score)
                chunk_score = 0
    return s

def eventloop(test_file):
    os.system('mkdir img')

    #96*54 pictures
    os.system('ffmpeg -y -i mov/%s -vf fps=6 -s 96x54 img/%s_%%d.png' % (test_file, test_file))

    #tv dataset
    _tv_file = open(test_file + '_tv_vmaf.log', 'w')
    #phone dataset
    _ph_file = open(test_file + '_ph_vmaf.log', 'w')

    tv_total_score = []
    ph_total_score = []
    os.system('ffmpeg -y -i mov/%s -strict -2 -c:v libx264 -b:v 2500k -maxrate 2500k -minrate 2500k -bufsize 2500k -f mp4 tmpor_%s' % (test_file, test_file))
    for b in BITRATE:
        os.system('ffmpeg -y -i mov/%s -strict -2 -c:v libx264 -b:v %dk -maxrate %dk -minrate %dk -bufsize %dk -f mp4 tmp_%s' % (test_file, b, b, b, b, test_file))
        #tv model
        os.system('../../../ffmpeg2vmaf %d %d tmpor_%s tmp_%s --out-fmt json 1>tmp_tv_%s.json' % (WIDTH, HEIGHT, test_file, test_file, test_file))
        #phone model
        os.system('../../../ffmpeg2vmaf %d %d tmpor_%s tmp_%s --phone-model --out-fmt json 1>tmp_ph_%s.json' % (WIDTH, HEIGHT, test_file, test_file, test_file))
        
        s = computer_score('tmp_tv_' + test_file + '.json')
        tv_total_score.append(s)
        s = computer_score('tmp_ph_' + test_file + '.json')
        ph_total_score.append(s)
    for col in range(len(tv_total_score[0])):
        for row in range(len(tv_total_score)):
            _tv_file.write(str(tv_total_score[row][col]))
            _tv_file.write(',')
        _tv_file.write('\n')
        _tv_file.flush()
    _tv_file.close()

    for col in range(len(ph_total_score[0])):
        for row in range(len(ph_total_score)):
            _ph_file.write(str(ph_total_score[row][col]))
            _ph_file.write(',')
        _ph_file.write('\n')
        _ph_file.flush()
    _ph_file.close()
    
    os.system('rm -rf tmp_%s' % (test_file))
    os.system('rm -rf tmpor_%s' % (test_file))
    print 'done'

if __name__ == '__main__':
    os.system('export PYTHONPATH=\"$(pwd)/../python/src:$PYTHONPATH\"')
    _total_file = os.listdir('mov/')
    _total_file.sort()
    for _file in _total_file:
        eventloop(_file)
