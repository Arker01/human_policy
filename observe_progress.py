import json,pathlib,time,sys,datetime
r=pathlib.Path(__file__).resolve().parents[1]
names=('visual_w01','visual_w1','readiness','geometry')
while True:
    complete=r/'results/completed.json'
    if complete.exists():
        print('训练及固定验证全部完成。',flush=True);sys.exit(0)
    p=r/'results/progress.json'
    if p.exists():
        s=json.load(open(p));parts=[]
        for n in names:
            row=s['training'].get(n,{});parts.append(n+': '+str(row.get('step',0))+'/100000')
            status=r/'results'/(n+'_status.json')
            if status.exists() and json.load(open(status))['exit_code']!=0:
                print('TRAINING_FAILED '+n,flush=True);sys.exit(2)
        print(datetime.datetime.now().strftime('%H:%M:%S')+' | '+'; '.join(parts)+' | 已验证 '+str(len(s['validated']))+'/4',flush=True)
    time.sleep(45)
