import java.util.*;
import java.lang.Math;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import javax.swing.*;


// This program demonstrates polynomial interpolation using Chebyshev nodes.
// It generates random y-values, computes corresponding Chebyshev x-values,
// merges nearby points, builds a Vandermonde matrix, and solves for the polynomial coefficients
// using the least squares method (via Gaussian elimination).
// It then checks lower-order formulas by moving down from the maximum order.
// The program then calculates and displays the interpolation error (RMSE),
// showing how Chebyshev nodes reduce oscillations and improve stability
// compared to evenly spaced points.
// Finally, it uses the JFreeChart library to plot these points and lines,
// and provide a better visual representation of the processes occuring. 
public class ChebyshevEngine{
	public static Random random;

	public static void main(String[] args){
		random = new Random();

		int numPoints = random.nextInt(7)+3; //Select random number of points to generate (between 3 and 9)
		double xMin=-1.0, xMax=1.0;
		double yMin=-1.0, yMax=1.0;

		double[] chebyshevNodes = computeChebyshevNodes(numPoints, xMin, xMax);
		System.out.println("Generated "+numPoints+" Chebyshev nodes between "+xMin+" and "+xMax+":");
		for(double x : chebyshevNodes){
			System.out.printf("  %.4f%n", x);
		}

		double[][] points = new double[numPoints][2];
		for(int i = 0; i<numPoints; i++){
			points[i][0] = chebyshevNodes[i];
			points[i][1] = yMin + (random.nextDouble() * (yMax-yMin));
		}
		System.out.println("\nAssigned random y-values to these nodes:");
		printPoints(points);

		System.out.println("\nMerging close points within a small threshold...");
		System.out.println("We now test progressively lower-degree polynomials to see if simpler models");
		System.out.println("can approximate the data nearly as well as the full high-order fit.");
		// Threshold can be adjusted
		double mergeThreshold = 0.1;
		// Merge points and output explanation
		points = mergeClosePoints(points, mergeThreshold);
		System.out.println("\nPoints after merging: (Points were sorted during merge)");
		printPoints(points);

		System.out.println("\nPerforming polynomial interpolation using the Least Squares Method...");
		double[] chebyshevCoefficients = polynomialInterpolation(points, chebyshevNodes);
		System.out.println("This solves (A^T A)a = (A^T y), where A is a Vandermonde matrix.");

		System.out.println("\nComparing Lower-Order Polynomials...");
		//Store all coefficient subsets and their errors
		int maxOrder = chebyshevCoefficients.length;
		java.util.List<double[]> lowerOrderCoefficients = new java.util.ArrayList<>();
		java.util.List<Double> lowerOrderErrors = new java.util.ArrayList<>();

		// Always include full order
		lowerOrderCoefficients.add(chebyshevCoefficients);
		lowerOrderErrors.add(calculateRMSE(chebyshevCoefficients, points));

		for (int order = maxOrder - 1; order >= 2; order--) {
		    double[] lowerCoeffs = java.util.Arrays.copyOf(chebyshevCoefficients, order);
		    double rmse = calculateRMSE(lowerCoeffs, points);
		    lowerOrderCoefficients.add(lowerCoeffs);
		    lowerOrderErrors.add(rmse);
		    System.out.printf("Polynomial Order %d -> RMSE = %.6f%n", order - 1, rmse);
		}

		// Identify smallest order within small tolerance of best fit
		double minError = lowerOrderErrors.get(0);
		int bestIndex = 0;
		for (int i = 1; i < lowerOrderErrors.size(); i++) {
		    if (Math.abs(lowerOrderErrors.get(i) - minError) / minError < 0.05) { // within 5%
		        bestIndex = i;
		    }
		}

		System.out.printf("Best simpler polynomial: degree %d (RMSE within 5%% of full model)%n",
		    lowerOrderCoefficients.get(bestIndex).length - 1);

		System.out.println("\nEvaluating polynomial at test points:");
		for(double x = -1.0; x<=1.0; x+=0.5){
			double yPred = 0.0;
			for(int j = 0; j<chebyshevCoefficients.length; j++){
				yPred += chebyshevCoefficients[j] * Math.pow(x, j);
			}
			System.out.printf("P(%.2f) = %.4f%n", x, yPred);
		}

		double chebyshevError = calculateInterpolationError(chebyshevCoefficients, chebyshevNodes, points);
		System.out.println("Chebyshev Interpolation Error: "+chebyshevError);
		System.out.println("(A smaller error indicates a better fit.)");

		System.out.println("\nSummary:");
		System.out.println("1. Random data was generated using Chebyshev nodes to ensure numerical stability.");
		System.out.println("2. Nearby points were merged to reduce noise.");
		System.out.println("3. Multiple polynomial degrees were tested; the simplest model within 5% RMSE was selected.");
		System.out.println("4. The graph visually compares these polynomial fits, showing the tradeoff between simplicity and accuracy.");
		System.out.println("End of program.");



		// Plot results using JFreeChart

		// Create a dataset
		XYSeries originalSeries = new XYSeries("Random Data (Points)");
		for(double[] point : points){
			originalSeries.add(point[0], point[1]);
		}

		XYSeries polynomialSeries = new XYSeries("Interpolated Polynomial");
		for(double x = -1.0; x<=1.0; x+=0.01){
			double yPred = 0.0;
			for(int j = 0; j<chebyshevCoefficients.length; j++){
				yPred += chebyshevCoefficients[j] * Math.pow(x, j);
			}
			polynomialSeries.add(x, yPred);
		}

		//Plot all polynomial orders and merged points

	//Create dataset for multiple polynomial fits
	XYSeriesCollection dataset = new XYSeriesCollection();

	// Plot each polynomial order (distinct label)
	for (int i = 0; i < lowerOrderCoefficients.size(); i++) {
	    double[] coeffs = lowerOrderCoefficients.get(i);
	    int degree = coeffs.length - 1;
	    XYSeries polySeries = new XYSeries("Polynomial Degree " + degree);

    	for (double x = -1.0; x <= 1.0; x += 0.01) {
    	    polySeries.add(x, evaluatePolynomial(coeffs, x));
    	}

    	if (i == bestIndex) {
        	polySeries = new XYSeries("Best Fit (Degree " + degree + ")");
        	for (double x = -1.0; x <= 1.0; x += 0.01) {
        	    polySeries.add(x, evaluatePolynomial(coeffs, x));
        	}
    	}
    	dataset.addSeries(polySeries);
	}

	// Plot merged data points
	XYSeries mergedSeries = new XYSeries("Merged Points");
	for (double[] p : points) {
	    mergedSeries.add(p[0], p[1]);
	}
	dataset.addSeries(mergedSeries);

	// Create chart
	JFreeChart chart = ChartFactory.createXYLineChart(
	        "Chebyshev Polynomial Fits (Multiple Orders)",
	        "x", "y", dataset,
	        PlotOrientation.VERTICAL,
	        true, true, false
	);

	org.jfree.chart.plot.XYPlot plot = chart.getXYPlot();
	org.jfree.chart.renderer.xy.XYLineAndShapeRenderer renderer =
	        new org.jfree.chart.renderer.xy.XYLineAndShapeRenderer();

	// Line styles (solid, dashed, dotted)
	java.awt.BasicStroke solid = new java.awt.BasicStroke(2.0f);
	java.awt.BasicStroke dashed = new java.awt.BasicStroke(
	        2.0f, java.awt.BasicStroke.CAP_BUTT, java.awt.BasicStroke.JOIN_BEVEL,
	        0, new float[]{6.0f, 6.0f}, 0);
	java.awt.BasicStroke dotted = new java.awt.BasicStroke(
	        2.0f, java.awt.BasicStroke.CAP_ROUND, java.awt.BasicStroke.JOIN_BEVEL,
	        0, new float[]{2.0f, 4.0f}, 0);

	// Assign distinct colors and styles to polynomial orders
	int numPolys = lowerOrderCoefficients.size();
	java.awt.Color[] colors = {
	        java.awt.Color.RED, java.awt.Color.BLUE, java.awt.Color.MAGENTA,
	        java.awt.Color.ORANGE, java.awt.Color.CYAN, java.awt.Color.PINK
	};
	java.awt.BasicStroke[] styles = { solid, dashed, dotted };

	for (int i = 0; i < numPolys; i++) {
	    renderer.setSeriesPaint(i, colors[i % colors.length]);
	    renderer.setSeriesStroke(i, styles[i % styles.length]);
	    renderer.setSeriesShapesVisible(i, false); // hide point markers for smooth lines
	}
	// Highlight best polynomial
	renderer.setSeriesStroke(bestIndex, new java.awt.BasicStroke(3.5f));
	renderer.setSeriesPaint(bestIndex, java.awt.Color.BLACK);

	// Last series = merged points -> show as green dots
	int mergedIndex = numPolys;
	renderer.setSeriesPaint(mergedIndex, java.awt.Color.GREEN);
	renderer.setSeriesStroke(mergedIndex, solid);
	renderer.setSeriesLinesVisible(mergedIndex, false); // points only
	renderer.setSeriesShapesVisible(mergedIndex, true);

	// Apply renderer to the plot
	plot.setRenderer(renderer);

	plot.setBackgroundPaint(java.awt.Color.WHITE);
	plot.setDomainGridlinePaint(java.awt.Color.LIGHT_GRAY);
	plot.setRangeGridlinePaint(java.awt.Color.LIGHT_GRAY);


	// Display chart
	JFrame frame = new JFrame("Chebyshev Polynomial Fits");
	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	frame.add(new ChartPanel(chart));
	frame.setSize(800, 600);
	frame.setVisible(true);


	}

	/**
	 * Performs polynomial interpolation using the Least Squares Method
	 * with Chebyshev nodes as the x-values and random data points as y-values.
	 * 
	 * This function builds and solves the equation:
	 * 		(A^T A)a = (A^T y)
	 * where A is a Vandermonde matrix constructed from the Chebyshev nodes,
	 * y is the vector of observed data values, and a is the resulting
	 * coefficient vector of the best-fit polynomial.
	 * 
	 * The resulting polynomial will be of degree n-1, where n is the number
	 * of data points. The use of Chebyshev nodes helps minimize Runge's
	 * phenomenon - large oscillations at the ends of the interpolation interval -
	 * making the approximation more stable than using evenly spaces points.
	 * 
	 * @param points A 2D array containing the original random (x,y) data points.
	 * @param chebyshevNodes An array of Chebyshev x-values mapped to the same range as the data.
	 * 
	 * @return An array of polynomial coefficients a0,a1,a2,...,an-1,
	 * 			where the resulting polynomial is P(x) = a0 + a1x + a2x^2 + ...
	 * 
	 * 
	 * Side Note: I used ChatGPT to help me write the documentation for this method.
	 * I wanted to make sure that I was using the correct terminology to describe the process.
	 **/
	public static double[] polynomialInterpolation(double[][] points, double[] chebyshevNodes){
		//Build Vandermonde matrix, A
		//For every data point x in points, plug its powers into A
		int n = points.length;
		double[][] A = new double[n][n];
		for(int i = 0; i<n; i++){
			for(int j = 0; j<n; j++){
				A[i][j] = Math.pow(chebyshevNodes[i], j);
			}
		}

		//Create vector with y values (right-hand side of system)
		double[] y = new double[n];
		for(int i = 0; i<n; i++){
			y[i] = points[i][1];
		}

		//Compute A^T A and A^T y
		double[][] ATA = multiplyTranspose(A);
		double[] ATy = multiplyTransposeVector(A, y);

		//Solve (A^T A)a = (A^T y)
		double[] coefficients = gaussianSolve(ATA, ATy);

		return coefficients;
	}

	/**
	 * Generates Chebyshev Nodes of the first kind, Chebyshev-Guass Nodes or Chebyshev zeros
	 * 
	 * @param numPoints Number of nodes to generate, also used as n in the formula
	 * @param xMin Lower bound of interpolation range
	 * @param xMax Upper bounds of interpolation range
	 * 
	 * @return double[] of length numPoints, containing ChebyshevNodes
	 **/
	public static double[] computeChebyshevNodes(int numPoints, double xMin, double xMax){
		double[] nodes = new double[numPoints];
		for(int i = 0; i<numPoints; i++){
			/* Formula for Nodes of the first kind:
				xi = cos((2i+1)*pi / 2n), i=0,1,...,n-1
			*/
			nodes[i] = 0.5*((xMin+xMax) + (xMax-xMin)*Math.cos((2*i+1)*Math.PI/(2*numPoints)));
		}

		return  nodes;
	}

	/**
	 * Merges points that are within a small x-range threshold of each other.
	 * 
	 * If two points have x-values closer than threshold, they are averaged
	 * into a single point. The method also prints a message explaining each merge.
	 * 
	 * This introduces a "slight error" (or correction) to the data by smoothing 
	 * small variations, reducing noise, and ensuring numerical stability
	 * 
	 * @param points The original array of (x, y) points.
	 * @param threshold The maximum allowed distance between two x-values before merging.
	 * @return A new array of merged (x, y) points.
	 **/
	public static double[][] mergeClosePoints(double[][] points, double threshold){
		//Sorts points by x to simplify comparison
		Arrays.sort(points, (a, b) -> Double.compare(a[0], b[0])); 

		List<double[]> mergedList = new ArrayList<>();

		int i = 0;
		while(i < points.length){
			double x = points[i][0];
			double y = points[i][1];

			//Check if next point is within threshold
			if(i < points.length -1 && Math.abs(points[i+1][0] - x) < threshold){
				double x2 = points[i+1][0];
				double y2 = points[i+1][1];

				double xAvg = (x+x2)/2;
				double yAvg = (y+y2)/2;

				System.out.printf("Merging points [%.4f, %.4f] and [%.4f, %.4f] -> new point [%.4f, %.4f]%n",
						x,y,x2,y2,xAvg,yAvg);

				mergedList.add(new double[] {xAvg, yAvg});
				i += 2; //Skip next point (since it's merged)
			} else{
				mergedList.add(new double[] {x,y});
				i++;
			}
		}

		//Convert list back to array
		double[][] mergedPoints = new double[mergedList.size()][2];
		for(int j = 0; j<mergedList.size(); j++){
			mergedPoints[j] = mergedList.get(j);
		}

		return mergedPoints;
	}


	/* Helper function using Gaussian elimination to solve a system of linear equations: A*x = b
		Where:
		A is a square matrix (A^T * A)
		b is a vector (A^T * y)
		Solving for x (coefficients of the polynomial)
	*/
	public static double[] gaussianSolve(double[][] A, double[] b){
		int n = A.length;

		//Forward elimination
		for(int p = 0; p<n; p++){
			//Pivot
			int max = p;
			for(int i = p+1; i<n; i++){
				if(Math.abs(A[i][p]) > Math.abs(A[max][p])){
					max = i;
				}
			}

			//Swap rows
			double[] temp = A[p];
			A[p] = A[max];
			A[max] = temp;
			double t = b[p];
			b[p] = b[max];
			b[max] = t;

			//Eliminate below
			for(int i = p+1; i<n; i++){
				double alpha = A[i][p] / A[p][p];;
				b[i] -= alpha * b[p];
				for(int j = p; j<n; j++){
					A[i][j] -= alpha*A[p][j];
				}
			}
		}

		//Back substitution
		double[] x = new double[n];
		for(int i = n-1; i>=0; i--){
			double sum = 0.0;
			for(int j = i+1; j<n; j++){
				sum += A[i][j] * x[j];
			}
			x[i] = (b[i] - sum)/A[i][i];
		}
		return x;
	}

	/**
	 * Calculates the interpolation error between a polynomial and a set of data points.
	 * 
	 * The error is measured as the root mean squared difference between the
	 * polynomial evaluated at the Chebyshev nodes and the actual y-values of
	 * the points. A smaller error indicates a better fit.
	 **/
	public static double calculateInterpolationError(double[] coefficients, double[] chebyshevNodes, double[][] points){
		int n = points.length;
		double errorSum = 0.0;

		//Evaluate the polynomial at each Chebyshev node and compare to actual y
		for(int i = 0; i<n; i++){
			double x = chebyshevNodes[i];
			double predictedY = 0.0;

			//Evaluate polynomial P(x) a0 + a1*x + a2*x^2 + ...
			for(int j = 0; j<coefficients.length; j++){
				predictedY += coefficients[j] * Math.pow(x,j);
			}

			double actualY = points[i][1];
			double diff = predictedY - actualY;
			errorSum += diff*diff; //accumulate squared differences
		}

		//Return the root mean squared error
		return Math.sqrt(errorSum / n);
	}

	//Calculates root mean squared error (RMSE) between predicted polynomial and actual data
	public static double calculateRMSE(double[] coefficients, double[][] points) {
    	double sum = 0.0;
    	for (double[] point : points) {
    	    double yPred = evaluatePolynomial(coefficients, point[0]);
    	    double diff = yPred - point[1];
    	    sum += diff * diff;
    	}
    	return Math.sqrt(sum / points.length);
	}

	//Helper function to multiply a matrix with its transpose
	public static double[][] multiplyTranspose(double[][] A){
		int n = A.length;
		int m = A[0].length;
		double[][] result = new double[m][m];

		for(int i = 0; i<m; i++){
			for(int j = 0; j<m; j++){
				double sum = 0;
				for(int k = 0; k<n; k++){
					sum += A[k][i] * A[k][j];
				}
				result[i][j] = sum;
			}
		}
		return result;
	}

	//Helper function to multiply a matrix transpose with a vector
	public static double[] multiplyTransposeVector(double[][] A, double[] y){
		int n = A.length;
		int m = A[0].length;
		double[] result = new double[m];

		for(int i = 0; i<m; i++){
			double sum = 0;
			for(int k = 0; k<n; k++){
				sum += A[k][i] * y[k];
			}
			result[i] = sum;
		}
		return result;
	}

	//Evaluates a polynomial P(x) = a0 + a1*x + a2*x^2 + ... for a given x
	public static double evaluatePolynomial(double[] coefficients, double x) {
    	double y = 0.0;
    	for (int j = 0; j < coefficients.length; j++) {
    	    y += coefficients[j] * Math.pow(x, j);
    	}
    	return y;
	}


	//Helper method to view randomly generated coordinates
	public static void printPoints(double[][] points){
		int n = points.length;
		for(int i = 0; i<n; i++){
			double x = points[i][0];
			double y = points[i][1];
			System.out.printf("[%.4f, %.4f]%n",x,y);
		}
	}
}
