
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
				value1[i] +=weight1[i][j]*sample[j];	
			}
			output1[i] = sigmoid(value1[i]);
		}
		
		for(int k=0; k<4; k++){
			value2 += weight1[4][k]*output1[k];
		}
		
		output2 = sigmoid(value2);
		error2 = target - output2;
		delta2 = output2*(1-output2)*error2;
		
		for(int l=0; l<4; l++){
			error1[l] = weight1[4][l]*delta2;
			delta1[l] = output1[l]*(1-output1[l])*error1[l];
		}
		
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
