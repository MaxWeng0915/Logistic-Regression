#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>


using namespace std;

struct DataRecord {
	double Age;
	double Weight;
	double BloodSugar;
	double Gender;
	double Diabetes;
};

int loadcsvfile2(string fname, map<string, vector<double>>& datas) {
	ifstream inputFile(fname);
	if (!inputFile.is_open()) {
		cerr << "Could not open the file" << endl;
		return 0;
	}

	string line;
	vector<DataRecord> data;

	if (getline(inputFile, line)) {
		while (getline(inputFile, line)) {
			stringstream ss(line);
			DataRecord record;
			char comma;
			ss >> record.Age >> comma;
			ss >> record.Weight >> comma;
			ss >> record.BloodSugar >> comma;
			string _Gender;
			getline(ss, _Gender, ',');
			if ("M" == _Gender)
				record.Gender = 1;
			else 
				record.Gender = 0;
			ss >> record.Diabetes;
			data.push_back(record);
		}
	}

	datas["Age"] = vector<double>();
	datas["Weight"] = vector<double>();
	datas["BloodSugar"] = vector<double>();
	datas["Gender"] = vector<double>();
	datas["Diabetes"] = vector<double>();

	for (const DataRecord& record : data) {
		datas["Age"].push_back(record.Age);
		datas["Weight"].push_back(record.Weight);
		datas["BloodSugar"].push_back(record.BloodSugar);
		datas["Gender"].push_back(record.Gender);
		datas["Diabetes"].push_back(record.Diabetes);
	}

	inputFile.close();
	return 1;
}

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double relu(double x) {
	return (x > 0) ? x : 0;
}

double compute_cost(vector<vector<double>> _x, vector<double> _y, vector<double> w, float b) {
	vector<double> y_pred;

	int size = w.size();
	for (int i = 0; i < _y.size(); i++) {

		double y = _y[i];
		double z = b;
		for (int j = 0; j < size; j++) {
			z += w[j] * _x[i][j];
		}	
		y_pred.push_back(sigmoid(z));


	}

	double sum = 0.0;
	for (int i = 0; i < _y.size(); i++) {
		double cost = -_y[i] * log(y_pred[i]) - (1 - _y[i]) * log(1 - y_pred[i]);
		sum += cost;
	}
	sum = sum / _y.size();

	return sum;
}

void compute_gradient(vector<vector<double>> _x, vector<double> _y, vector<double> w, float b, vector<float>& _w_gradient, float& _b_gradient) {
	int size = w.size();
	vector<float> sumw(size, 0.0);
	float sumb = 0.0;
	for (int i = 0; i < _x.size(); i++) {

		float y = _y[i];
		vector<float> w_gradients(size, 0.0);

		// y_pred = w1 * x1 + ... + wn *xn + b
		float y_pred = b;
		for (int j = 0; j < size; j++) {
			y_pred += w[j] * _x[i][j];
		}
		y_pred = sigmoid(y_pred);

		for (int j = 0; j < size; j++) {
			w_gradients[j] = _x[i][j] * (y_pred - y);
			sumw[j] += w_gradients[j];
		}
		float b_gradient = (y_pred - y);
		sumb += b_gradient;
	}

	for (int j = 0; j < size; j++)
		_w_gradient[j] = sumw[j] / _x.size();
	_b_gradient = sumb / _x.size();
}



double calculateMean(const vector<double>& data) {
	double sum = 0.0;
	for (double value : data) {
		sum += value;
	}
	return sum / data.size();
}

double calculateStandardDeviation(const vector<double>& data) {
	double mean = calculateMean(data);
	double sumOfSquaredDifferences = 0.0;

	for (double value : data) {
		double difference = value - mean;
		sumOfSquaredDifferences += difference * difference;
	}

	double variance = sumOfSquaredDifferences / data.size();
	double standardDeviation = sqrt(variance);

	return standardDeviation;
}


vector<double> featureScaling(vector<double> data) {
	double mean = calculateMean(data);
	double sd = calculateStandardDeviation(data);
	for (int i = 0; i < data.size(); i++)
		data[i] = (data[i] - mean) / sd;

	return data;

}

int main()
{
	map<string, vector<double>> datas;
	int res = loadcsvfile2("Diabetes_Data.csv", datas);
	if (res == 1) {

		datas["Age"] = featureScaling(datas["Age"]);
		datas["Weight"] = featureScaling(datas["Weight"]);
		datas["BloodSugar"] = featureScaling(datas["BloodSugar"]);
		datas["Gender"] = featureScaling(datas["Gender"]);


		vector<double> w = { 1, 2, 3, 4 };
		float b = 1.0;
		vector<vector<double>> x;
		for (int i = 0; i < datas["Age"].size(); i++)
		{
			vector<double> x_sub;
			x_sub.push_back(datas["Age"][i]);
			x_sub.push_back(datas["Weight"][i]);
			x_sub.push_back(datas["BloodSugar"][i]);
			x_sub.push_back(datas["Gender"][i]);
			x.push_back(x_sub);

			/*cout << "i: " << i << " Age: " << datas["Age"][i] << endl;
			cout << "i: " << i << " Weight: " << datas["Weight"][i] << endl;
			cout << "i: " << i << " BloodSugar: " << datas["BloodSugar"][i] << endl;
			cout << "i: " << i << " Gender: " << datas["Gender"][i] << endl;*/
		}

		//double cost = compute_cost(x, datas["Diabetes"], w, b);		
		//cout << " Cost: " << cost << endl;

		float learning_rate = 0.1;
		float cost;

		for (int i = 0; i < 20000; i++) {
			vector<float> w_gradient = { 0.0, 0.0, 0.0, 0.0 };
			float b_gradient;
			compute_gradient(x, datas["Diabetes"], w, b, w_gradient, b_gradient);

			w[0] = w[0] - w_gradient[0] * learning_rate;
			w[1] = w[1] - w_gradient[1] * learning_rate;
			w[2] = w[2] - w_gradient[2] * learning_rate;
			w[3] = w[3] - w_gradient[3] * learning_rate;
			b = b - b_gradient * learning_rate;

			cost = compute_cost(x, datas["Diabetes"], w, b);

			if (i % 1000 == 0)
				cout << "i: " << i << " Cost: " << cost << " w[0]:" << w[0] << " b:" << b << " w_gradient[0]:" << w_gradient[0] << " b_gradient:" << b_gradient << endl;
		}		

		
	}


	system("pause");
	return 0;
}