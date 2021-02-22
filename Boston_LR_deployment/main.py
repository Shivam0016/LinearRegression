from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle

app=Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def index():
    if request.method=='POST':
        try:
            crime=float(request.form['crime'])
            residential=float(request.form['residential'])
            non_retail=float(request.form['non_retail'])
            river=float(request.form['river'])
            nitric=float(request.form['nitric'])
            rooms=float(request.form['rooms'])
            owner_occupied=float(request.form['owner_occupied'])
            e_distance=float(request.form['e_distance'])
            highways=float(request.form['highways'])
            tax=float(request.form['tax'])
            pt_ratio=float(request.form['pt_ratio'])
            bbt=float(request.form['bbt'])
            lower_status=float(request.form['lower_status'])

            f1='scaler.pickle'
            f2='poly_feat.pickle'
            f3='poly_mod.pickle'

            scaler=pickle.load(open(f1,'rb'))
            poly_feat=pickle.load(open(f2,'rb'))
            poly_mod=pickle.load(open(f3,'rb'))

            scaled_data=scaler.transform([[crime,residential,non_retail,river,nitric,rooms,owner_occupied,e_distance,highways,tax,pt_ratio,bbt,lower_status]])
            polynomial_features=poly_feat.transform(scaled_data)
            predicted_value=poly_mod.predict(polynomial_features)

            print(crime)
            print(residential)
            print(non_retail)
            print(river)
            print(nitric)
            print(rooms)
            print(owner_occupied)
            print(e_distance)
            print(highways)
            print(tax)
            print(pt_ratio)
            print(bbt)
            print(lower_status)
            print(scaled_data)
            print(polynomial_features)
            print(predicted_value)

            return render_template('result.html',predictions=round(1000*predicted_value[0]))
        except Exception as e:
            print('The error message is ',e)
            return render_template('index.html')
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run()
