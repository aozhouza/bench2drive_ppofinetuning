import json
path="/home/aozhou/bench2drive/Bench2Drive-main/uniad_b2d_traj/eval_bench2drive220_0_uniad_20route.json"
with open(path,'r') as file:
    data=json.load(file)
res=data["_checkpoint"]["records"]
ans1=0
ans2=0
ans3=0
for dic in res:
    ans1+=dic['scores']["score_route"]
    ans2+=dic['scores']["score_penalty"]
    ans3+=dic['scores']["score_composed"]
print(ans1/len(res),ans2/len(res),ans3/len(res))
