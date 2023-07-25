import numpy as np 
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def init_data():
    with open('train.bin','rb') as f1:
        train = pickle.load(f1)
     

    with open('test.bin','rb') as f2 :
        test = pickle.load(f2)
    return train, test

def data_ready1():# 여기서는 300개 어레이 10개로 이루어진 세트 나중에 100개만 가져와야하나
    trainSet =[]
    testSet =[]
    for i in range(10):
        trainSet.append(train[i][0:testNo])#첫번째 인덱스는 숫자, 두번째 인덱스는 각 숫자별 번쨋수, 세번째는 인덱스 이런 3중구조인덧
        testSet.append(test[i][0:testNo])
    return trainSet, testSet

def data_ready2(train, test, trainNo,testNo):#k fold 안하나? 300 100
    trainSetf=np.zeros((trainNo*10, 28))#28*28 짜리 300개는 하나의 
    testSetf=np.zeros((testNo*10, 28))
    #np.concatenate((np.sum(train[i][j] ,axis=0), np.sum(train[i][j] ,axis=0)))

    for i in range(len(train)): # 아 10개구나
        for j in range(trainNo):#300
            trainSetf[i*trainNo+j,:] =  np.sum(train[i][j] ,axis=0) #x 축 y 축 순서.
    for i in range(len(test)):
        for j in range(testNo):
            testSetf[i*testNo+j,:] =  np.sum(train[i][j] ,axis=0)
    return trainSetf,testSetf
    #3000개 1000개. 0 300개, 1 300개 2 300개 이렇게 일렬로 나열된 형태. 
def createTmpl(trainSet): #트레인셋 하나를 넣어주면 10개 한줄로 붙인 템플릿 반환. 트레인셋에 각각의 숫자 이미지 전부 평균내서
    tmpl = np.zeros((1,28*10))
    #tmpl = np.zeros((1,10,28))
    for i in range(10):
        imsi = np.array(trainSet[i])# 마찬가지 1부터 9까지 숫자별로 300개 3중, train i 는 각 숫자별 300개가 됨
        tmpl[:,i*28:(i+1)*28] = np.mean(imsi, axis = 0)#아마 두번째 인덱스에 해당하는 것, 즉 28*2x28*2짜리를 원소로 평균낸듯
        #10 * 100* 28 => 100*28(image)
        #print('tmpl shape',tmpl.shape)
    return tmpl

def templeteMatch(tmpl,testSet):# testNO?
    result = np.zeros((testNo,10))
    print(testNo)
    for i in range(len(testSet)):# 0부터 9 까지 순회
        for n in range(len(testSet[i])):# 각 숫자마다 100개씩 이미지 순회
            #print(testSet[i][n].shape)
            imsiTest = np.tile(testSet[i][n],(1,10))# 그럼 그 각 이미지 하나를 베껴 10개를 복사해서 한줄로 합쳐
            #print(imsiTest.shape)
            #
            error = np.abs(tmpl - imsiTest)#템플릿이랑 (10개) 비교하는거 = 뺄셈 abs
            errorSum = [error[:,0:28].sum(),error[:,28:28*2].sum(),
                        error[:,28*2:28*3].sum(),error[:,28*3:28*4].sum(),     #즉 겹치지 않는 픽셀, 그리고 합쳤으니까 그들의 갯수
                        error[:,28*4:28*5].sum(),error[:,28*5:28*6].sum(),
                        error[:,28*6:28*7].sum(),error[:,28*7:28*8].sum(),
                        error[:,28*8:28*9].sum(),error[:,28*9:28*10].sum()]# 각각의 템플릿과의 에러 저장
            result[n,i] = np.argmin(errorSum) # 에러가 가장 적은 것 result에 저장. 
                                            #인덱스 반환하는 꼼수로 약간 컨퓨전 매트릭스 그림 생각해보면됨 
                                            #가로축이 각 수, 새로축이 인덱스, 좌표의 값은 에러 가장 적었던 템플릿
    return result# 모든 행의 i 번째 원소들은 사실 같은 수이다, 단 템플릿 매치 결과 뭐와 가장 차이가 적게 났는가가 기록되어 있다.

def calcMeasure(result):
#acc= (tp+tn)/ (tp+fn+fp+tn)
# pre =tp/ (tp+fp)
# rec =tp/(tp+fn)
# f1 = 2*pre*rec/(pre+rec)
    label = np.tile(np.arange(0,10),(testNo,1))#testno 수만큼의 원소안에 1개씩 10개짜리 어레이 넣는다는 소린데
    confusionMat = []
    
    TP, TN, FP, FN = [],[],[],[]#1 5 15 100 200 500 가 아니다 그건 k고 데이터 셋 크기는 300 100고정임
    for i in range(10):
        TP.append(((result == label) & (label ==i)).sum())#둘이 같은 좌표에 같은 값이 있고, 그 값이 i 일 경우. 계산 다 이런식임.
        TN.append(((result !=i) & (label !=i)).sum())
        FP.append(((result != label) & (label ==i)).sum())
        FN.append(((result ==i) & (label !=i)).sum())
        #(testNo/10)*i:(testNo/10)*(i+1)
    
    for i in range(10):
        imsi = []
        for n in range(10):
        #imsi.append((result[(testNo/10)*n:(testNo/10)*(n+1),n]).sum())
            imsi.append(result[:,i].tolist().count(n))
        
        confusionMat.append(imsi)

    confusionMat = np.array(confusionMat)
    TP =np.array(TP)
    TN =np.array(TN) 
    FN =np.array(FN) 
    FP =np.array(FP)
    acc= (TP+TN)/(TP+TN+FP+FN)
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    f1 = 2*pre*rec/(pre+rec)
    
    return acc, pre, rec, f1,confusionMat

def show():
    train, test = init_data()
    for i in range(10):
        plt.subplot(2,5,i+1) 
        plt.imshow(train[i][0],'gray')
        plt.axis('off')
        print(len(train[i]))

    plt.show()

def templateMatchRun():
    global testNo ,testNum
    global train, test
    train, test = init_data()
 
    global trainNo 
    trainNo, testNo = 300, 1000

    testEa =[10, 50, 100, 200, 300, 500, 1000]
    f1_list =[]

    for i in range(len(testEa)):
        testNum = testEa[i]
        trainSet , testSet = data_ready2(train, test,trainNo,testNo)
        trainSet =trainSet.reshape(10,300,28)
        testSet = testSet.reshape(10,testNo,28)
        templ =createTmpl(trainSet)
        
        result_1 = templeteMatch(templ,testSet)
        
        acc, pre, rec, f1, confusionMat = calcMeasure(result_1)
        f1_mean = np.mean(f1)
        f1= np.insert(f1,0,f1_mean)
        f1_list.append(f1)

    df = pd.DataFrame(f1_list)
    df.to_csv('HistTmpl2.csv', index=False)


def knn(trainSet,testSet, k):
    #숫자별로 300개씩 있는데 그거 전부 해보는듯. 1 300개 2 300개 이렇게 테스트셋 이미지 하나당 
    sizeTr=int(trainSet.shape[0]/10)#얘 둘의 모양은 2차원 이미지를 flatten 하듯이 만든거라고 생각하면 편함.
    sizeTe=int(testSet.shape[0]/10) #300, 100 이미지 개수와 동일
    result=np.zeros((sizeTe,10))

    for i in range(testSet.shape[0]):#0부터 28*28 까지 아닌가? 이 축은 1000개 있을텐데 testSet[0] / testSet.shape[0]
        imsi=np.sum((trainSet-testSet[i,:])**2,axis=1)# 그럼 3000 * 1 짜리 행렬,28*28 대신 1, 안겹치는 픽셀수
        
        
        #테스트셋에서 하나 뽑아서 그걸 트레인셋에 감산한다는게 어떻게 되는건지. 
        no=np.argsort(imsi)[0:k]# 가장 값이 작은것 k 개 (의 인덱스)
        hist, bins=np.histogram(no//sizeTr,np.arange(-0.5,10,1))# //300, 300으로 나눈 정수 나누기 몫, 즉 그 수가 몇인지, 숫자들의 분포표
        result[i%sizeTe,i//sizeTe] = np.argmax(hist)#몇번째 클래스의 몇번째 데이터의 결과는 가장 이미지가 비슷한 것  k개 중 가장 많이 분포
        #0부터 100까지 각 행은 10개 100개의 숫자중
    return result#100 * 10   result 0~100 중 몇번째 이미지의 0~9까지 중 어디냐



testNo = 0
trainNo = 0
train, test = [],[]

def knnRun():
    global testNo 
    global trainNo 
    global train, test

    train , test = init_data()
    trainNo, testNo = 300, 100

    testEa =[5]#k 1,5,15,100,200,500
    f1_list =[]
    
    trainSet , testSet = data_ready2(train, test,trainNo,testNo)
    
    for i in range(len(testEa)):
        
        rsult_2 = knn(trainSet , testSet , testEa[i])
        
        acc, pre, rec, f1, confusionMat = calcMeasure(rsult_2)#여러 클래스 분류중이라 f1은 평균을 냄.
        f1_mean = np.mean(f1)
        f1= np.insert(f1,0,f1_mean)# 맨 앞에 f1 평균.
        f1_list.append(f1)
    
    f1_list.append(np.zeros(10))
    acc_mean = []

#k fold 일단 트레인 셋에서 300개식 10숫자 3000개, 5개로 나눠서 함
#코드 상에서 제일 결과 좋은 거 빼고 지울까 했는데 그냥 보고 지우는걸로
    trainNo, testNo = 240, 60#중요, 지우면 코드 많이 고쳐야 됨.
    for n in range(5): #위에건 k 값 바꿔가며, 이거는 5fold
        
        index = np.arange(n*60,(n+1)*60)
        trainSet2 =trainSet.reshape(10,300,28*2)#delete 함수 쓰기 위해 모양 바꿈.
        
        testSet2 = trainSet2[:,n*60:(n+1)*60]#60개 추출
        trainSet2 = np.delete(trainSet2, index ,axis = 1)#60개 제외

        trainSet2 = trainSet2.reshape(2400,28*2)#원래대로 
        testSet2 = testSet2.reshape(600,28*2)
        
        rsult_2 = knn(trainSet2 , testSet2, 5)

        acc, pre, rec, f1, confusionMat = calcMeasure(rsult_2)
        f1_mean = np.mean(f1)
        f2 = np.zeros(11)
        f2= np.insert(f1,0,f1_mean)
        acc_mean.append(acc)
        f1_list.append(f2)
    f1_list.append(np.zeros(11))#엑셀기준 표 밑에 한줄 0으로 구분.
    #f1_list.append(confusionMat)
    #f1_list.append(np.zeros(10))
    acc_mean = np.array(acc_mean)
    acc_mean = np.mean(acc_mean, axis = 0)
    acc_mean = np.insert(acc_mean,0,'133')
    f1_list.append(acc_mean)#참고로 acc도 각 클래스별로 하나씩 나오는 비율임.
    
    f1_list.append(np.zeros(10))
    f1_list.append(confusionMat)

    df = pd.DataFrame(f1_list)
    df.to_csv('HistKnn2.csv', index=False)
    print('knn done')
    
    
#knnRun()
templateMatchRun()
#show()
#아 기억한게 맞네용 각 300개씩 총 3000개 5fold 감사합니다
#머신러닝2 5fold k값은 1-500개 돌린거중에 best값으로 하면된답니다.
#그리고 머신러닝2는 3000개 중 600개 테스트 5 fold로 하라고 하셨던 것 같습니다! 숫자별로 60개 10개인거구만
#그러니까 각 300 개 0부터 9까지 근데 60개씩 나눠서 5번 k fold 이거지?