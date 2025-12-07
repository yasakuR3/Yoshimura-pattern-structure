import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 一部環境では不要だけど一応
import trimesh

def koten(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):

    ox = x2 - x1
    oy = y2 - y1
    oz = z2 - z1

    hx = x4 - x3
    hy = y4 - y3
    hz = z4 - z3

    wx = x1 - x3
    wy = y1 - y3
    wz = z1 - z3

    a = ox ** 2 + oy ** 2 + oz ** 2
    b = ox * hx + oy * hy + oz * hz
    c = hx ** 2 + hy ** 2 + hz ** 2
    d = ox * wx + oy * wy + oz * wz
    e = hx * wx + hy * wy + hz * wz

    s = (b * e - c * d) / (a * c - b ** 2)
    t = (a * e - b * d) / (a * c - b ** 2)

    px = x1 + s * ox
    py = y1 + s * oy
    pz = z1 + s * oz

    qx = x3 + t * hx
    qy = y3 + t * hy
    qz = z3 + t * hz

    return (px + qx) / 2.0, (py + qy) / 2.0, (pz + qz) / 2.0

def hosen_bekutoru(x1, y1, z1, x2, y2, z2, x3, y3, z3, u):

    nx = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    ny = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
    nz = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    heihokon = np.sqrt(nx**2+ny**2+nz**2)

    return u * (nx / heihokon), u * (ny / heihokon), u * (nz / heihokon)

def draw_points_sets(point_sets, a):
    # ある面
    p1 = point_sets[0]
    p2 = point_sets[1]
    p3 = point_sets[2]

    v1, v2, v3 = hosen_bekutoru(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], a)

    p10 = p1[0] + v1
    p10i = p1[0] - v1
    p11 = p1[1] + v2
    p11i = p1[1] - v2
    p12 = p1[2] + v3
    p12i = p1[2] - v3

    p20 = p2[0] + v1
    p20i = p2[0] - v1
    p21 = p2[1] + v2
    p21i = p2[1] - v2
    p22 = p2[2] + v3
    p22i = p2[2] - v3

    p30 = p3[0] + v1
    p30i = p3[0] - v1
    p31 = p3[1] + v2
    p31i = p3[1] - v2
    p32 = p3[2] + v3
    p32i = p3[2] - v3

    # ある面
    p4 = point_sets[3]
    p5 = point_sets[4]
    p6 = point_sets[5]
    
    g1, g2, g3 = hosen_bekutoru(p4[0], p4[1], p4[2], p5[0], p5[1], p5[2], p6[0], p6[1], p6[2], a)

    p40 = p4[0] + g1
    p40i = p4[0] - g1
    p41 = p4[1] + g2
    p41i = p4[1] - g2
    p42 = p4[2] + g3
    p42i = p4[2] - g3

    p50 = p5[0] + g1
    p50i = p5[0] - g1
    p51 = p5[1] + g2
    p51i = p5[1] - g2
    p52 = p5[2] + g3
    p52i = p5[2] - g3

    p60 = p6[0] + g1
    p60i = p6[0] - g1
    p61 = p6[1] + g2
    p61i = p6[1] - g2
    p62 = p6[2] + g3
    p62i = p6[2] - g3

    xm1, ym1, zm1 = koten(p10, p11, p12, p30, p31, p32, p40, p41, p42, p60, p61, p62)
    xm1i, ym1i, zm1i = koten(p10i, p11i, p12i, p30i, p31i, p32i, p40i, p41i, p42i, p60i, p61i, p62i)

    xm2, ym2, zm2 = koten(p20, p21, p22, p30, p31, p32, p50, p51, p52, p60, p61, p62)
    xm2i, ym2i, zm2i = koten(p20i, p21i, p22i, p30i, p31i, p32i, p50i, p51i, p52i, p60i, p61i, p62i)

    return xm1, ym1, zm1, xm2, ym2, zm2, xm1i, ym1i, zm1i, xm2i, ym2i, zm2i

def draw_edges(ax, origamis, edges):
    """
    edges: [ ((h1, k1, face1, i1), (h2, k2, face2, i2)), ... ]
    の形のリスト
    """
    for (h1, k1, f1, i1), (h2, k2, f2, i2) in edges:
        p = origamis[h1, k1, f1, i1, :]
        q = origamis[h2, k2, f2, i2, :]

        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
        )

def build_mesh_from_triangles(origamis, triangles):
    """
    triangles: [ ((h1,k1,f1,i1), (h2,k2,f2,i2), (h3,k3,f3,i3)), ... ]
    という形のリスト
    """
    vertices = []
    faces = []
    vertex_map = {}

    for tri in triangles:
        face = []
        for key in tri:
            if key not in vertex_map:
                h, k, f, i = key
                x, y, z = origamis[h, k, f, i, :]
                vertex_map[key] = len(vertices)
                vertices.append([x, y, z])
            face.append(vertex_map[key])
        faces.append(face)

    vertices = np.array(vertices, dtype=float)
    faces = np.array(faces, dtype=int)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh

def main():
    # 設計変数
    a = 150
    b = 80

    # 三角形の個数
    n = 12 # 偶数
    n1 = n // 2
    # 開き角度
    r = 75

    # 上に重ねる個数 [個数]
    k = 4

    # 折り紙の厚さ[mm] 
    t = 0.6 # 1.0[mm]以下推奨 厚いと歪な形状になる。3Dプリンタが印刷可能な最低厚さを要確認

    # ★ origamis 配列を用意 (厚さを考慮した点、上に重ねる個数、1ユニットの二つの面、ある面における点の数、　xyz座標)
    origamis = np.zeros((3, k, 2, n1, 3), dtype=float)

    origamis[0, 0, 0, 0, 0] = (a / 2.0) / np.tan(np.radians(360/n))
    origamis[0, 0, 0, 0, 1] = a / 2.0
    origamis[0, 0, 0, 0, 2] = 0

    origamis[0, 0, 1, 0, 0] = origamis[0, 0, 0, 0, 0] + a * np.cos(np.radians(r))
    origamis[0, 0, 1, 0, 1] = 0
    origamis[0, 0, 1, 0, 2] = a * np.sin(np.radians(r))

    for j in range(0, 2):
        for i in range(1, n1):
            angle = np.radians(720/n)
            x_prev = origamis[0, 0, j, i-1, 0]
            y_prev = origamis[0, 0, j, i-1, 1]
            origamis[0, 0, j, i, 0] = x_prev * np.cos(angle) - y_prev * np.sin(angle)
            origamis[0, 0, j, i, 1] = x_prev * np.sin(angle) + y_prev * np.cos(angle)
            origamis[0, 0, j, i, 2] = origamis[0, 0, j, 0, 2]
    
    # 厚さを考慮した場合の点の座標計算
    # 面A
    pts1 = origamis[0, 0, 0, 0, :]
    pts2 = origamis[0, 0, 1, 0, :]
    pts3 = origamis[0, 0, 0, n1-1, :]

    # 面B
    ptv1 = origamis[0, 0, 0, 0, :]
    ptv2 = origamis[0, 0, 1, 0, :]
    ptv3 = origamis[0, 0, 1, 1, :]

    point_sets1 = [pts1, pts2, pts3, ptv1, ptv2, ptv3]

    xm1, ym1, zm1, xm2, ym2, zm2, xm1i, ym1i, zm1i, xm2i, ym2i, zm2i = draw_points_sets(point_sets1, t/2.0)

    origamis[1, 0, 0, 0, 0] = xm1
    origamis[1, 0, 0, 0, 1] = ym1
    origamis[1, 0, 0, 0, 2] = zm1

    origamis[1, 0, 1, 0, 0] = xm2
    origamis[1, 0, 1, 0, 1] = ym2
    origamis[1, 0, 1, 0, 2] = zm2

    origamis[2, 0, 0, 0, 0] = xm1i
    origamis[2, 0, 0, 0, 1] = ym1i
    origamis[2, 0, 0, 0, 2] = zm1i

    origamis[2, 0, 1, 0, 0] = xm2i
    origamis[2, 0, 1, 0, 1] = ym2i
    origamis[2, 0, 1, 0, 2] = zm2i

    for h in range(1, 3):
        for j in range(0, 2):
            for i in range(1, n1):
                angle = np.radians(720/n)
                x_prev = origamis[h, 0, j, i-1, 0]
                y_prev = origamis[h, 0, j, i-1, 1]
                origamis[h, 0, j, i, 0] = x_prev * np.cos(angle) - y_prev * np.sin(angle)
                origamis[h, 0, j, i, 1] = x_prev * np.sin(angle) + y_prev * np.cos(angle)
                origamis[h, 0, j, i, 2] = origamis[h, 0, j, 0, 2]
    
    # 上に重ねる部分の座標計算
    # ★ origamis 配列を用意 (厚さを考慮した点、上に重ねる個数、1ユニットの二つの面、ある面における点の数、　xyz座標)
    if k >= 2:
        for g in range(1, 3):
            for j in range(0, n1):
                origamis[g, 1, 1, j, 0] = origamis[g, 0, 0, j, 0]
                origamis[g, 1, 1, j, 1] = origamis[g, 0, 0, j, 1]
                origamis[g, 1, 1, j, 2] = origamis[g, 0, 0, j, 2] + 2 * origamis[g, 0, 1, 0, 2]     
    
    for g in range(1, 3):
        for i in range(2, k):
            for j in range(0, n1):
                origamis[g, i, 1, j, 0] = origamis[g, i-2, 1, j, 0]
                origamis[g, i, 1, j, 1] = origamis[g, i-2, 1, j, 1]
                origamis[g, i, 1, j, 2] = origamis[g, i-2, 1, j, 2] + 2 * origamis[g, 0, 1, 0, 2]   

    triangles = []

    for i in range(0, n1):
        # 底面A
        triangles.append(((1, 0, 0, i), (2, 0, 0, i), (2, 0, 0, (i+1)%n1)))
        triangles.append(((1, 0, 0, (i+1)%n1), (1, 0, 0, i), (2, 0, 0, (i+1)%n1)))

        # 上面B
        triangles.append(((1, k-1, 1, i), (2, k-1, 1, i), (2, k-1, 1, (i+1)%n1)))
        triangles.append(((1, k-1, 1, (i+1)%n1), (1, k-1, 1, i), (2, k-1, 1, (i+1)%n1)))

        triangles.append(((1, 0, 0, (i+1)%n1), (1, 0, 0, i), (1, 0, 1, (i+1)%n1)))
        triangles.append(((2, 0, 0, (i+1)%n1), (2, 0, 0, i), (2, 0, 1, (i+1)%n1)))

        triangles.append((((1, 0, 0, i), (1, 0, 1, i), (1, 0, 1, (i+1)%n1))))
        triangles.append((((2, 0, 0, i), (2, 0, 1, i), (2, 0, 1, (i+1)%n1))))
    
    for i in range(1, k):
        for j in range(0, n1):
            if i % 2 == 1:
                triangles.append(((1, i, 1, j), (1, i-1, 1, j), (1, i-1, 1, (j+1)%n1)))
                triangles.append(((1, i, 1, j), (1, i, 1, (j+1)%n1), (1, i-1, 1, (j+1)%n1)))

                triangles.append(((2, i, 1, j), (2, i-1, 1, j), (2, i-1, 1, (j+1)%n1)))
                triangles.append(((2, i, 1, j), (2, i, 1, (j+1)%n1), (2, i-1, 1, (j+1)%n1)))

            else:
                triangles.append(((1, i-1, 1, j), (1, i-1, 1, (j+1)%n1), (1, i, 1, (j+1)%n1)))
                triangles.append(((1, i-1, 1, j), (1, i, 1, j), (1, i, 1, (j+1)%n1)))

                triangles.append(((2, i-1, 1, j), (2, i-1, 1, (j+1)%n1), (2, i, 1, (j+1)%n1)))
                triangles.append(((2, i-1, 1, j), (2, i, 1, j), (2, i, 1, (j+1)%n1)))

    mesh = build_mesh_from_triangles(origamis, triangles)
    mesh.export("yoshimura-pattern.stl")  # カレントディレクトリに出力

    # ======= ここから「線だけ」を描画する部分 =======
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 全点（軸のスケール計算用。描画はしない）
    points = origamis.reshape(-1, 3)

    edges = []

    for g in range(1, 3):
        # for i range(0, k):
        for j in range(0, n1):
            edges.append(((g, 0, 0, j), (g, 0, 0, (j+1)%n1)))
            edges.append(((g, 0, 1, j), (g, 0, 1, (j+1)%n1)))
            edges.append(((g, 0, 0, j), (g, 0, 1, j)))
            edges.append(((g, 0, 0, (j+1)%n1), (g, 0, 1, j)))

    # 線を描画
    draw_edges(ax, origamis, edges)

    # 軸ラベルとスケール調整
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    xyz_min = min(xs.min(), ys.min(), zs.min())
    xyz_max = max(xs.max(), ys.max(), zs.max())
    ax.set_xlim(xyz_min, xyz_max)
    ax.set_ylim(xyz_min, xyz_max)
    ax.set_zlim(xyz_min, xyz_max)

    plt.show()

if __name__ == "__main__":
    main()
    
