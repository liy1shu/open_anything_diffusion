import os

lines = []
dir = "/home/yishu/open_anything_diffusion/logs/sim_policies_trajectory_diffuser_pndit/2024-05-06/02-43-41/logs/simu_eval"
for filename in os.listdir(dir):
    if ".html" not in filename:
        continue
    path = os.path.join(dir, filename)
    with open(path, "r") as f:
        new_lines = []
        for line in f.readlines():
            new_lines.append(
                line.replace("http://128.2.178.238:9002/", "http://128.2.178.238:9010/")
            )

    with open(path, "w") as f:
        f.writelines(new_lines)
