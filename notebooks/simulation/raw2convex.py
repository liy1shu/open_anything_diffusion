from tqdm import tqdm

ids = [
    "8867",
    "8877",
    "8893",
    "8897",
    "8903",
    "8919",
    "8930",
    "8961",
    "8997",
    "9016",
    "9032",
    "9035",
    "9041",
    "9065",
    "9070",
    "9107",
    "9117",
    "9127",
    "9128",
    "9148",
    "9164",
    "9168",
    "9277",
    "9280",
    "9281",
    "9288",
    "9386",
    "9388",
    "9410",
    "8983",
    "8994",
    "9003",
    "9263",
    "9393",
]

# # Part - 1: obj files
# for object_id in tqdm(ids):
#     obj_path = f"/home/yishu/datasets/partnet-mobility/convex/{object_id}/textured_objs/"
#     file_names = [path for path in os.listdir(obj_path) if '.obj' in path]
#     for file_name in tqdm(file_names):
#         p.connect(p.DIRECT)
#         name = f"/home/yishu/datasets/partnet-mobility/convex/{object_id}/textured_objs/{file_name}"
#         name_log = "log.txt"
#         p.vhacd(name, name, name_log)

# Part - 2: urdf
for object_id in tqdm(ids):
    urdf_path = (
        f"/home/yishu/datasets/partnet-mobility/convex/{object_id}/mobility.urdf"
    )
    new_lines = []
    with open(urdf_path, "r") as f:
        for line in f.readlines():
            new_lines.append(line.replace('<collision concave="yes">', "<collision>"))

    # new_urdf_path = f"/home/yishu/datasets/partnet-mobility/convex/{object_id}/new_mobility.urdf"
    with open(urdf_path, "w") as f:
        f.writelines(new_lines)
