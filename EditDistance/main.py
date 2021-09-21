import numpy as np

deleteCost = 1
insertCost = 1
replaceCost = 2


def getDistTable(srcStr, dstStr):
    srcLen = len(srcStr)
    dstLen = len(dstStr)

    dist = np.zeros((srcLen + 1, dstLen + 1, 2))
    for i in range(srcLen + 1):
        dist[i, 0, 0] = i * deleteCost
        dist[i, 0, 1] = 0
    for j in range(dstLen + 1):
        dist[0, j, 0] = j * insertCost
        dist[0, j, 1] = 1
    dist[0, 0, 1] = -1

    for i in range(1, srcLen + 1):
        for j in range(1, dstLen + 1):
            cost = [dist[i - 1, j, 0] + deleteCost, dist[i, j - 1, 0] + insertCost]
            if srcStr[i - 1] == dstStr[j - 1]:
                cost.append(dist[i - 1, j - 1, 0])
            else:
                cost.append(dist[i - 1, j - 1, 0] + replaceCost)
            minPos = np.argmin(cost)
            dist[i, j, 0] = cost[minPos]
            dist[i, j, 1] = minPos
            if cost[2] == cost[minPos]:
                dist[i, j, 1] = 2
    return dist


def askDist(srcEnd, dstEnd, dist):
    srcPos = srcEnd
    dstPos = dstEnd
    path = []
    while dist[srcPos, dstPos, 1] >= 0:
        way = dist[srcPos, dstPos, 1]
        path.append(way)
        if way == 0:
            srcPos = srcPos - 1
        elif way == 1:
            dstPos = dstPos - 1
        else:
            srcPos = srcPos - 1
            dstPos = dstPos - 1
    return dist[srcEnd, dstEnd, 0], path


def printPath(srcStr, dstStr, path):
    srcPos = len(srcStr)
    dstPos = len(dstStr)
    for way in path:
        if way == 0:
            print("删除第" + str(srcPos) + "位字符" + srcStr[srcPos - 1])
            srcPos = srcPos - 1

        elif way == 1:
            print("在第" + str(srcPos + 1) + "位插入字符" + dstStr[dstPos - 1])
            dstPos = dstPos - 1
        else:
            if srcStr[srcPos - 1] == dstStr[dstPos - 1]:
                print("原始串第" + str(srcPos) + "位字符与目标串第" + str(dstPos) + "位字符" + srcStr[srcPos - 1] + "匹配")
            else:
                print("替换第" + str(srcPos) + "位字符" + srcStr[srcPos - 1] + "为" + dstStr[dstPos - 1])

            srcPos = srcPos - 1
            dstPos = dstPos - 1
        print(srcStr[:srcPos] + dstStr[dstPos:])


if __name__ == "__main__":
    srcStr = input("原始串：")
    dstStr = input("目标串：")
    dist = getDistTable(srcStr, dstStr)
    print("距离矩阵：")
    print(dist[:, :, 0])

    srcEnd = int(input("原始串前缀长度："))
    dstEnd = int(input("目标串前缀长度："))
    cost, path = askDist(srcEnd, dstEnd, dist)
    print("最小编辑距离：", cost)
    printPath(srcStr[:srcEnd], dstStr[:dstEnd], path)
