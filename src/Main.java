
public class Main {

	public static void main(String[] args) {
		
		double[][] sample = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
		double[][] weight1 = {{0.3, 0.2, 0.1}, {0.5, 0.2, 0.6}, {0.8, 0.2, 0.4}, {0.7, 0.4, 0.2}, {0.5, 0.5, 0.5, 0.5}};
		//double[] weight2 = {0.5, 0.5, 0.5, 0.5};
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
