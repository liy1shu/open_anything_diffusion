import os

lines = []
dir = "/home/yishu/open_anything_diffusion/logs/sim_eval_trajectory_diffuser_dgdit/2024-03-25/04-07-46/logs/simu_eval"
for filename in os.listdir(dir):
    if ".html" not in filename:
        continue
    path = os.path.join(dir, filename)
    with open(path, "r") as f:
        new_lines = []
        for line in f.readlines():
            new_lines.append(
                line.replace("http://128.2.178.238:9000/", "http://128.2.178.238:9003/")
            )

    with open(path, "w") as f:
        f.writelines(new_lines)
