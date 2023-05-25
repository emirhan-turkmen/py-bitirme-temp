import cv2
import numpy as np
import utlis

# girdiler
questions = utlis.questions()
ans = utlis.cevaplari_kaydet(questions)
webcamFeed = utlis.webcamFeed()
cameraNo = 1
if webcamFeed == True:
    cameraNo = utlis.camSelect()

########################
path = "1.png"
widthImg = 600
heightImg = 700
# questions = 5  bu soru sayısı değiştirilebilir kullanıcıdan alsak veya oto belirlesek süper olur
choices = 5 # 5 yapınca 5 işaretleme 10 yapınca 10 işaretleme yapıyor ama yerleri yanlış ve 5 ten sonra hepsi doğru gözüküyor
# ans = [0, 1, 3, 2, 1, 3, 2, 4, 0, 1] # cevaplar kullanıcıdan alınacak
# webcamFeed = True # cam olup olmayacağını bu belirliyor
# cameraNo = 1 # bu bilgisayarda 1 laptop cam, 0 telefona bağlanan uygulamaya bağlı
########################

cap = cv2.VideoCapture(cameraNo)
cap.set(10, 150)


while True:
    if webcamFeed: success, img = cap.read()
    else: img = cv2.imread(path)

    # preprossesing
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    try:
        # finding all contours
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)
        # find rectangles
        recCon = utlis.rectContour(contours)
        biggestContour = utlis.getCornerPoints(recCon[0])
        gradePoints = utlis.getCornerPoints(recCon[1])
        #print(biggestContour)

        if biggestContour.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBiggestContours, biggestContour, -1, (0,255,0), 20)
            cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

            biggestContour = utlis.reorder(biggestContour)
            gradePoints = utlis.reorder(gradePoints)

            # ansvers
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # grades
            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
            # cv2.imshow("Grade", imgGradeDisplay)

            # APPLY THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            '''
            burada soru sayısı mı şık sayısı mı alıyoruz emin değilim sorun olabilir
            olası çözüm: 2. bi parametre olarak questions veririz     
            '''
            boxes = utlis.splitBoxes(imgThresh, questions) # questions parametresi eklendi
            # cv2.imshow("Test", boxes[2])
            # print(cv2.countNonZero(boxes[1]), cv2.countNonZero(boxes[2]))

            # getting nonzero pixel values of each box
            myPixelVal = np.zeros((questions, choices))
            countC = 0
            countR = 0
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if (countC == choices): countR += 1 ;countC = 0
            # print(myPixelVal)

            # finding index values of markings
            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                # print("arr", arr)
                myIndexVal = np.where(arr == np.amax(arr))
                # print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])
            # print(myIndex)


            # grading
            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else: grading.append(0)
            # print(grading)
            score = (sum(grading)/questions) * 100 # final grade
            print(score)

            utlis.save_results(score)

            # grading dizisi doğru
            # myIndex dizisi doğru

            # displaying answers
            imgResults = imgWarpColored.copy()
            imgResults = utlis.showAnswers(imgResults, myIndex, grading, ans, questions, choices)
            imgRawDrawing = np.zeros_like(imgWarpColored)
            imgRawDrawing = utlis.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)
            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))



            # grade show
            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade, str(int(score)) + "", (100, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (255,255,255), 3)
            # cv2.imshow("grade", imgRawGrade)
            InvMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, InvMatrixG, (widthImg, heightImg))



            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)




        imgBlank = np.zeros_like(img) # temp
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgContours, imgBiggestContours, imgWarpColored, imgThresh],
                      [imgResults, imgRawDrawing, imgInvWarp, imgFinal])
    except:
        imgBlank = np.zeros_like(img) # temp
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgBlank, imgBlank, imgBlank, imgBlank],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    lables = [["Original", "Gray", "Blur", "Canny"],
              ["Contours", "Biggest Con.", "Warp", "Threshold"],
              ["Result", "Raw Drawing", "Inv Warp", "Final"]]
    imageStacked = utlis.stackImages(imageArray, 0.3, lables)

    cv2.imshow("final result", imgFinal)
    cv2.imshow("Stacked Images",imageStacked)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("FinalResult.jpg", imgFinal)
        cv2.waitKey(300)