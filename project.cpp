#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <math.h> 

/*

	g++ project.cpp -o project `pkg-config --cflags --libs opencv`
	,/project

*/

using namespace cv;
using namespace std;

/*
** Függvény amivel kiszámoljuk a bal felső és jobb alsó sarkát a táblázatnak
** Meghatározzuk a kontúrokat, és a CHAIN_APPROX_SIMPLE segítségével csak 
** pár pontot határozunk meg a kontúrból. A bal felső lesz az a pont, ami a 
** legközelebb van a bal felső sarokhoz, a jobb alsó pont lesz az, ami legtávolabb
** van a bal felső saroktól.
*/

void calculate_corners(Mat img, Point& UpperLeft, Point& BottomRight) {

	double min_dist = 50000, max_dist = 0;

	vector<vector<Point>> contours;
	findContours(img, contours, RetrievalModes::RETR_EXTERNAL, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

	for(int i = 0; i < contours.size(); i++){
		for(int k = 0; k < contours[i].size(); k++){
			if(pow(contours[i][k].y, 2) + pow(contours[i][k].x, 2) < min_dist){
				min_dist = pow(contours[i][k].y, 2) + pow(contours[i][k].x, 2);
				UpperLeft = contours[i][k];
			}
			if(pow(contours[i][k].y, 2) + pow(contours[i][k].x, 2) > max_dist){
				max_dist = pow(contours[i][k].y, 2) + pow(contours[i][k].x, 2);
				BottomRight = contours[i][k];
			}
		}
	}

}

/*
** A táblázat két széle által bezárt derékszögű háromszög segítségével 
** kiszámoljuk a háromszög oldalait és szögeit nagyjából ~
** Ezek alapján megkapjuk azt, hogy mennyivel van elforgatva az ábra 
** az eredeti ábrához képest.
*/

void calculate_abc(double& a, double& b, double& c, Point A, Point B, Point C, double& alpha, double& beta){

	a = sqrt(pow((B-C).x, 2) + pow((B-C).y, 2));
	b = sqrt(pow((A-C).x, 2) + pow((A-C).y, 2));
	c = sqrt(pow((B-A).x, 2) + pow((B-A).y, 2));

	alpha = 180 * asin(a / c) / 3.14159265359;
	beta = 180 * asin(b / c) / 3.14159265359;

}

/*
** A megadott képre kirajzolja a háromszög 3 pontját.
** ellenőrző jellegű függvény
*/

void drawCircles(Mat& src, Point A, Point B, Point C){

	circle(src, A, 4, Scalar(255, 0, 0), 3);
	circle(src, C, 4, Scalar(0, 0, 255), 3);
	circle(src, B, 4, Scalar(0, 255, 0), 3);

}

/*
** A kép elforgatása megfelelő szöggel
** StackOverflow-s
*/

void rotation(double angle, Mat src, Mat& dst){

	// A kép középpontjának meghatározása
    Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
	
	// Az elforgatáshoz tartozó mátrix kialakítása
    Mat rot = getRotationMatrix2D(center, angle, 1.0);

	//Az elforgatott kép mérete
    Rect2f bbox = RotatedRect(Point2f(), src.size(), angle).boundingRect2f();

	/*
	** A transzformációs mátrix 2. oszlopát állítja be úgy, hogy 
	** a kép középpontja azonos helyre kerüljön.
	*/
		rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
		rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

    warpAffine(src, dst, rot, bbox.size());

}

/*
** Itt eltűntetjuk a kisebb lyukakat, tehát az üres cellák 
** megmaradnak, és azok tűnnek el, amikben van jelölés, mivel
** a jelölés kissebb lyukakra osztja a cellát.
*/

void fillHoles(Mat& src){

	Mat kernel = getStructuringElement(MORPH_RECT, Size(35, 37));
	dilate(src, src, kernel);
	erode(src, src, kernel);

}

/*
** A függvény már egy lyukak nélküli képet kap meg, azon
** végigszalad, és ha talál egy fekete pontot, akkor megkeresi
** a ponthoz tartozó feket négyszög átellenes pontját, és ezek 
** alapján meg tudjuk határozni a négyszög közepét.
** A négyszögeket egy "kétdimenzió" vektorban tároljuk, 
** az y koordinátájuk alapján. Ezekkel később egyszerű 
** lesz meghatározni, hogy az adott sorban lévő értékek
** megfelelnek-e a megoldáson található sorral.
** Ha találunk (found == true) fekete pontot a sorban
** akkor, amikor végigmentünk a soron nem eggyel kell 
** növelni az i-t, hanem a négyszög aljától egy kicsivel 
** lentebb kell folytatni. Ugyanígy a j-vel, csak azt 
** a soron belül többször eltolni.
*/

vector<vector<Point>> detectLines (Mat& src){

	vector<vector<Point>> lines;

	int k, l;
	int counter = 0;
	bool found = false;
	for (int i = 0; i < src.rows; i++) {
			found = false;
		for (int j = 0; j < src.cols; j++) {

			if (src.at<uchar> (i, j) == 0) {
				k = i + 2;
				l = j + 2;
				while(src.at<uchar>(k, l) == 0){
					k++;
				}
				k-=2;
				while(src.at<uchar>(k, l) == 0){
					l++;
				}
				l-=2;

			//Ha a 0. négyszög kisebb a határétéknél akkor fejjel lefele van a kép
				if(counter == 0 && (i - k) * (j - l) < 1600){

					rotation(180, src, src);

					return detectLines(src);
				}
				
				found = true;
				counter++;

				bool inLine = false;

				for(int m = 0; m < lines.size(); m++){
					if(lines[m][0].y + 10 > (i + k) / 2.0 && lines[m][0].y - 10 < (i + k) / 2.0){
						inLine = true;
						lines[m].push_back(Point((j + l) / 2.0, (i + k) / 2.0));
						break;
					}
				}

				if(!inLine){
					vector<Point> temp;
					temp.push_back(Point((j + l) / 2.0, (i + k) / 2.0));
					lines.push_back(temp);
				}

				j = l + 10;
			}
		}
		if(found){
			i = k+10;
		}
	}

	return lines;
}

int main(int argvc, char* argv[]){

/*
** Szükséges változók deklarálása
** ABC pontok a táblázat 2 széléhez, és a bezárt derékszögű háromszög 
** 	harmadik pontjához
** abc számok az oldalak hosszaihoz
** alpha beta számok a háromszög két másik szögének
**
** Mátrixok: 
** 	th		- küszöbölt kép
**  bgr		- BGR színskálás
** 	cropped	- kivágott
*/

	Point A, B, C;
	double a, b, c, alpha, beta;
	vector<vector<Point>> solverRects;

	Mat test_solver, test_solver_th, test_solver_bgr, test_solver_th_cropped;

/*
** Alapkép beolvasása
*/

	test_solver = imread("tesztek/test_solver.png", IMREAD_GRAYSCALE);

	if(test_solver.empty()){
		cerr << "Didn't find test_solver.png... " << endl;
		return -1;
	}	
	
	imshow("test_solver", test_solver);
	
/*
** Küszöbölés
*/

	threshold(test_solver, test_solver_th, 150, 255, THRESH_BINARY_INV);

	//imshow("test_solver_th", test_solver_th);

/*
** A küszöbölt képen bal felső és jobb alsó pont meghatározása
** majd ezek alapján a két pontot derékszögű háromszögre kiegészítő
** pont kiszámolása
*/

	calculate_corners(test_solver_th, A, B);
	C = Point(A.x, B.y);

/*
** A háromszög szögeinek és oldalainak kiszámolása
*/

	calculate_abc(a, b, c, A, B, C, alpha, beta);

/*
** A háromszög pontok kirajzolása a táblázatra ellenőrző jelleggel
*/

	cvtColor(test_solver, test_solver_bgr, COLOR_GRAY2BGR);
	drawCircles(test_solver_bgr, A, B, C);

	//imshow("test_solver_bgr", test_solver_bgr);
	
	//cout << "a: " << a << "\tAlpha diff: \t" << alpha-alpha << endl;
	//cout << "b: " << b << "\tBeta diff: \t" << beta-beta << endl;
	//cout << "c: " << c << endl << endl;

/*
** A kép kivágása, majd átméretezése
** Rect(x, y, width, height) 
*/

	Rect cutShape(A.x-2, A.y-2, B.x - A.x + 5, B.y - A.y + 5);

	test_solver_th_cropped = test_solver_th(cutShape);
	resize(test_solver_th_cropped, test_solver_th_cropped, Size(test_solver_th_cropped.cols - 2, test_solver_th_cropped.rows - 2));
	//imshow("test_solver_th_cropped", test_solver_th_cropped);

/*
** A kisebb lyukak, azaz a jelölt cellák eltűntetése
*/

	fillHoles(test_solver_th_cropped);
	//imshow("test_solver_th_cropped_no_holes", test_solver_th_cropped);

/*
** A lyuk nélküli képen megmaradt négyszögek középpontjainak 
** a detectálása, és soronkénti tárolása
*/

	solverRects = detectLines(test_solver_th_cropped);
	waitKey();

	for(int n = 0; n <= 21; n++){
	
	/*
	** Szükséges változók létrehozása, ami itt extra: 
	** points			- minden képhez tartozik egy pontszám
	** percentage	- mennyi százalékban adott helyest választ
	** Mat-> rot	-	elforgatott kép
	*/

		vector<vector<Point>> testRects;
		int points = 0;
		double percentage = 0;

		Point A_, B_, C_;
		double a_, b_, c_, alpha_, beta_;

		Mat test, test_th, test_bgr, test_th_rot, test_th_cropped, test_bgr_rects;

	/*
	** A solver képnél alkalmazott lépések
	*/

		test = imread("tesztek/test_" + to_string(n) + ".png", IMREAD_GRAYSCALE);

		if(test.empty()){
			cerr << "Didn't find test_" << n << ".png... " << endl;
			continue;
		}	

		imshow("test", test);

		threshold(test, test_th, 150, 255, THRESH_BINARY_INV);
		//imshow("test_th", test_th);

		calculate_corners(test_th, A_, B_);
		C_ = Point(A_.x, B_.y);

		calculate_abc(a_, b_, c_, A_, B_, C_, alpha_, beta_);

		cvtColor(test, test_bgr, COLOR_GRAY2BGR);
		drawCircles(test_bgr, A_, B_, C_);

		//cout << "a: " << a_ << "\tAlpha diff: \t" << alpha_ - alpha << endl;
		//cout << "b: " << b_ << "\tBeta diff: \t" << beta_ - beta << endl;
		//cout << "c: " << c_ << endl << endl;
		//imshow("test_bgr", test_bgr);

	/*
	** A beolvasott képek elforgatása, hogy a solvernek
	** megfelelően legyen forgatva. Ha az alpha nagyobb 
	** mint a beta akkor fekvő képünk van, ezért 
	** elforgatjuk állóvá, majd javítjuk a forgatást
	*/

		test_th_rot = test_th.clone();

		if(alpha_ > beta){
			rotation(alpha_ + alpha_ - beta, test_th, test_th);
			calculate_corners(test_th, A_, B_);
			C_ = Point(A_.x, B_.y);

			calculate_abc(a_, b_, c_, A_, B_, C_, alpha_, beta_);
		}

		rotation(beta_ - beta, test_th, test_th_rot);

		//imshow("test_rot", test_th_rot);


	/*
	** Pontok, hosszak újraszámolása
	*/

		calculate_corners(test_th_rot, A_, B_);
		C_ = Point(A_.x, B_.y);

		calculate_abc(a_, b_, c_, A_, B_, C_, alpha_, beta_);

		cvtColor(test_th_rot, test_bgr, COLOR_GRAY2BGR);
		drawCircles(test_bgr, A_, B_, C_);

	/*
	** A képből csak a táblázat kivágása
	*/

		Rect cut_sample(A_.x, A_.y, B_.x - A_.x, B_.y - A_.y);
		test_th_cropped = test_th_rot(cut_sample);

	/*
	** Átméretezés a solver kép méreteinek megfelelően
	*/

		resize(test_th_cropped, test_th_cropped, Size(test_solver_th_cropped.cols, test_solver_th_cropped.rows));
		("test_th_cropped", test_th_cropped);

	/*
	** Valamiért megjelentek szürke pontok is a képen ezért újra 
	** küszöböltem
	*/

		threshold(test_th_cropped, test_th_cropped, 150, 255, THRESH_BINARY);

	/*
	** A bejelölt cellák eltüntetése
	*/

		fillHoles(test_th_cropped);
		//imshow("test_th_cropped_no_holes", test_th_cropped);

	/*
	** A bgr_rects képen lesznek színes pontok a helyes sorokon, 
	** ezért bgr színtérbe áttesszük a képet, majd a négyszögek detectálása
	*/

		testRects = detectLines(test_th_cropped);
		cvtColor(test_th_cropped, test_bgr_rects, COLOR_GRAY2BGR);

	/*
	** A megoldáson lévő négyszögsávhoz megkeressük az teszten lévő 
	** megfelelő magasságban lévő sávot, és ha ugyanannyi cella van 
	** jelölés nélkül akkor belép, egyébként továbblép és arra a 
	** a sorra 0 pontot kap automatikusan az illető
	*/

		for(int i = 1; i < solverRects.size(); i++){
			for(int k = 1; k < testRects.size(); k++){
				if(solverRects[i][0].y - 10 < testRects[k][0].y && solverRects[i][0].y + 10 > testRects[k][0].y){
					if(solverRects[i].size() == testRects[k].size()){
						int count = 0;

					/*
					** Végigmegyünk a solver i-edik során, és keresünk
 					** a test k-adik sorában olyat, ami megfelel az adott
					** i-edik sorban és j-edik oszlopban lévő négyszög x
					** koordinátájának, ha van ilyen akkor a számlálót 
					** megnöveljük
					*/

						for(int j = 0; j < solverRects[i].size(); j++){
							for(int l = 0; l < testRects[k].size(); l++){
								if(solverRects[i][j].x - 10 < testRects[k][l].x && solverRects[i][j].x + 10 > testRects[k][l].x){
									count++;
								}
							}
						}

					/*
					** Ha annyi négyszöget találtunk amennyi van a 
					** megoldás soraiban, akkor adunk egy pontot a 
					** tesztre (illetve ellenőrző jelleggel lerakunk
					** zöld köröket a helyes sor üres celláira
					*/

						if(count == solverRects[i].size()){
							for(int l = 0; l < testRects[k].size(); l++){
								circle(test_bgr_rects, testRects[k][l], 7, Scalar(0, 200, 0), 14);
							}
								points++;
						}
					}
				}
			}
		}

	/*
	** Pontszám kiíratása, és az összefirkált kép megmutatása
	*/

		cout << n << ". teszt: " << points << " pont\t" << 100*points / (double)(solverRects.size() - 1) << "%" << endl;
		imshow("test_bgr_rects ", test_bgr_rects);

		waitKey();
	}

	return 0;
}
