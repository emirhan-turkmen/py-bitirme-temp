import cv2
import numpy as np

def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,0,0),2)
    return ver

def rectContour(contours):

    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def reorder(myPoints):

    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    #print(myPoints)
    #print(add)
    myPointsNew[0] = myPoints[np.argmin(add)] # origin point [0, 0]
    myPointsNew[3] = myPoints[np.argmax(add)] # [w, h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] # [w, 0]
    myPointsNew[2] = myPoints[np.argmax(diff)] # [0, h]
    #print(diff)

    return myPointsNew


'''
rows = np.vsplit(img, 5) bu kısımda 5 soru için hali bunu 
fonksiyona soru sayısı için bir parametre vererek
çözebiliriz sanırım mainde questions değişkenini bu fonksiyona
rows olarak verdiğimizde kaç soru istersek cevaplanrıabiliriz
img yanına bir questions parametresi ile çözülebilir
'''
def splitBoxes(img, questions): # questions parametresi eklendi
    rows = np.vsplit(img, questions) #sorun olabilir satır sayısı bunu parametre olarak verebiliriz
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)

    # cv2.imshow("split", rows[0])
    return boxes

def showAnswers(img, myIndex, grading, ans, questions, choices):
    secW = int(img.shape[1]/questions)
    secH= int(img.shape[0]/choices)

    for x in range(0, questions):
        myAns = myIndex[x]
        # finding box center for right markings
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2

        if grading[x] == 1:
            myColor = (0, 255, 0) # green true
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
        else:
            myColor = (0, 0, 255) # red false
            correctAns = ans[x]

            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

            # correct ans
            myColor = (0, 255, 0)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW) + secW//2, (x * secH) + secH//2),
            20, myColor, cv2.FILLED)


        #cv2.circle(img, (int(cX), int(cY)), 50, myColor, cv2.FILLED)
    return img

def questions():
    questions = int(input("soru sayısını giriniz: "))
    return questions

def cevaplari_kaydet(questions):
    ans = []

    for i in range(questions):
        soru = f"{i + 1}. Soru: "
        cevap = input(soru)

        cevap = cevap.upper()
        secenekler = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

        if cevap in secenekler:
            cevap_sayi = secenekler[cevap]
            ans.append(cevap_sayi)
        else:
            print("Geçersiz şık! Lütfen A, B, C, D veya E şıklarından birini girin.")
            return []

    return ans

def webcamFeed():
    webcamFeed = str(input("web cam ile canlı okuma yapılacak mı? evet/hayır: "))
    if webcamFeed == "evet":
        return True
    elif webcamFeed == "hayır":
        return False
    else:
        print("hatalı seçim yapıldı")
        exit(1)

def camSelect():
    print("genelde 1 varsayılan 0 sonradan eklenen olur")
    camSelect = int(input("kullanacağınız kamera nedir: "))
    return camSelect

def save_results(score):
    with open("sonuclar.txt", "a") as file:
        file.write(f"1, {score}\n")