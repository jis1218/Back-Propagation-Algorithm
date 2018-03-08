#Back Propagation Algorithm
##### > Multi Layer Neural Network의 한 종류로 은닉 노드가 여러개이다.
##### > 코드에서 구현된 알고리즘의 모식도는 다음과 같다.

![이미지](/img/BackPropagation.png)

```java
public class BackPropagation {
	
	public double[][] doBackPropa(double[] sample, double[][] weight1, double target){
		
		double alpha = 0.8;
		
		double[] value1 = new double[4];
		double[] output1 = new double[4];
		double value2 = 0;
		double output2 = 0;
		double error2 = 0;
		double delta2 = 0;
		double[] error1 = new double[4];
		double[] delta1 = new double[4];
		
		for(int i=0; i<4; i++){
			for(int j=0; j<3; j++){				
				value1[i] +=weight1[i][j]*sample[j];  //각 Sample의 합성곱 값을 구함
			}
			output1[i] = sigmoid(value1[i]); //Sigmoid 함수에 합성곱 값을 넣어줌
		}
		
		for(int k=0; k<4; k++){
			value2 += weight1[4][k]*output1[k]; // Sigmoid 함수에 합성곱 값을 넣어 나온 값과 weight을 곱해서 출력 노드의 v값을 구함
		}
		
		output2 = sigmoid(value2); // 최종 v 값을 Sigmoid 함수에 넣어줌
		error2 = target - output2; // 출력 노드의 에러값을 구한다.
		delta2 = output2*(1-output2)*error2; //에러값을 이용해 최종노드의 델타값을 구한다.
		
		for(int l=0; l<4; l++){
			error1[l] = weight1[4][l]*delta2; // 은닉노드 error 값 정의에 의해 은닉노드의 에러값과 델타값을 구할 수 있음
			delta1[l] = output1[l]*(1-output1[l])*error1[l];
		}
		
		//weight을 갱신
		for(int m=0; m<4; m++){
			for(int n=0; n<3; n++){
				weight1[m][n] += alpha*delta1[m]*sample[n]; 
			}
			weight1[4][m] += alpha*delta2*output1[m];
		}
		
		return weight1;
	}	
	
	public double sigmoid(double v){		
		return 1/(1+Math.exp(-1*v));	
	}
}
```
```java
public class Main {

	public static void main(String[] args) {
		
		double[][] sample = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
		double[][] weight1 = {{0.3, 0.2, 0.1}, {0.5, 0.2, 0.6}, {0.8, 0.2, 0.4}, {0.7, 0.4, 0.2}, {0.5, 0.5, 0.5, 0.5}};
		double[] target = {0, 1, 1, 0};
		
		BackPropagation backPropagation = new BackPropagation();
		
		for(int m=0; m<100000; m++){
			for(int i=0; i<sample.length; i++){
				weight1 = backPropagation.doBackPropa(sample[i], weight1, target[i]);
			}
		}		
		
		double output[] = new double[4];
		
		
		// output 출력하는 함수
		for(int k=0; k<sample.length; k++){
			double sum[] = new double[4];
			double fisum = 0;
			for(int j=0; j<4; j++){
				for(int y=0; y<3; y++){
					sum[j] += weight1[j][y]*sample[k][y];				
				}	
				output[j] = backPropagation.sigmoid(sum[j]);
				fisum +=output[j]*weight1[4][j];
			}			
			double result = backPropagation.sigmoid(fisum);
			System.out.println(result);			
		}	
	}
}
```

##### 단층 신경망과는 달리 target 값이 {0, 1, 1, 0} 일때 실제값도 비슷하게 얻을 수 있다.
##### 하지만 여전히 값이 {0.3, 0.4, 0.2, 0.7} 이렇게 되는 경우 잘 안나온다. 왜 그런것일까??