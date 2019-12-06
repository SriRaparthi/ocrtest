import indent as indent
import ocrconfig as cfg
import requests,json
import argparse
import mysql.connector
from flask import Flask, request


indent.msg2 = None
indent.ktp_msg = None
indent.selfie_msg = None

app = Flask(__name__)
#if __name__ == "__main__":
#    app.run(host='0.0.0.0',port = '5000')


@app.route("/welcomeocr", methods=['POST','GET'])
def hello_world():
   return "Hello, World!"

def retjson(ocr_json):
    python2json = json.dumps(ocr_json)
    return python2json

@app.route("/ocrmain", methods=['POST','GET'])
def main():
# def main(file_path, file_type, ap_temp_id):
    # app.run(host='10.15.15.85', port=5000, debug=True)
    content = request.get_json(force = True)
    file_path = content[0]["path"]
    print(file_path, end=" -> ");print(type(file_path))
    file_type = content[0]["type"]
    ap_temp_id = content[0]["tempId"]
    #PATH_TO_IMAGE,PATH_TO_SELFIE_IMAGE --> previous parameters
    # CONNECT TO DB
    configsource = cfg.dbconn
    cnx = mysql.connector.connect(**configsource)
    mycursor = cnx.cursor()
    if(cnx):
        print("connected to mysql database")
    else:
        print("OCR server unable to connect to DB")

    if file_type == 'ktp':
        PATH_TO_IMAGE = file_path
        image,indent.msg2 = indent.getimage(PATH_TO_IMAGE)
        if indent.msg2 == 'KTP Image file does not exist':
            raise Exception(indent.msg2)
        #image read succesfully... and sending for processing
        ktpscore,ktpresult,indent.ktp_msg = indent.ktpimageproc(image)
        try:    
            if ktpscore is None:
                ktpscore = "NULL"
            #ap_temp_id = 1099 #static for now
            #ocr_decision = "NULL" #static for now.
            ocr_error = ""
            
            if ktpresult == "PASS":
                #KTP test is success
                # ocr_error = "\""+'NULL'+"\""
                ocr_error = "NULL"
            
            elif ktpresult == "FAIL":
                #KTP test is Fail
                ocr_error = "\""+indent.ktp_msg+"\""
                print(ocr_error)
            
            # else:
            #     print("Invalid KTP Result")
            #     print(ktpresult)
            #     response_json = {'result':ktpresult, 'ocr_error': indent.ktp_msg }
            
            dbcheckquery = "SELECT * FROM ocr_data WHERE ocr_ap_id_temp = {} ORDER BY ocr_id desc LIMIT 1".format(ap_temp_id)
            mycursor.execute(dbcheckquery)
            dbcheckres = mycursor.fetchall()
            if mycursor.rowcount > 0:
                #update query goes here
                updatequery = "UPDATE ocr_data SET ocr_ktp_score = "+str(ktpscore)+", ocr_ktp_result = "+"\""+ktpresult+"\""+", ocr_error = "+ocr_error+" WHERE ocr_ap_id_temp = "+str(ap_temp_id) + " ORDER BY ocr_id desc LIMIT 1"
                mycursor.execute(updatequery)
                cnx.commit()
                print("Row already exists updating")
        
            else:
                insertquery = "INSERT INTO ocr_data (ocr_ap_id_temp, ocr_ktp_score, ocr_ktp_result, ocr_error,ocr_raw_response) VALUES (%s, %s ,%s, %s, %s)"
                ocr_raw_response = {'ocr_ap_id_temp': ap_temp_id, 'ocr_ktp_result':ktpresult,'ocr_error' : ocr_error}
                insertion_json = {'ocr_ap_id_temp': ap_temp_id, 'ocr_ktp_score':ktpscore, 'ocr_ktp_result':ktpresult,'ocr_error' : ocr_error,'ocr_raw_response' : str(ocr_raw_response)}
                val = tuple(insertion_json.values())
                mycursor.execute(insertquery, val)
                cnx.commit()
                print("Inserted into database")
            
            #Closing the connection
            cnx.close()
            response_json = {'result':ktpresult, 'ocr_error': ocr_error}
            return retjson(response_json)
        
        except Exception as e:
            cnx.rollback()
            print("Exception caught Reverting SQL changes")
            #Closing the connection
            cnx.close()
            ktpresult = "FAIL"
            print(e)
            response_json = {'result':ktpresult, 'ocr_error': str(e) }
            return retjson(response_json)
        
    elif file_type == 'selfie':
        try:
            PATH_TO_SELFIE_IMAGE = file_path
            selfie_image,indent.msg2 = indent.getselfie_img(PATH_TO_SELFIE_IMAGE)
            if indent.msg2 == 'Selfie Image file does not exist':
                raise Exception(indent.msg2)
            selfiescore,selfieresult,indent.selfie_msg = indent.selfieimageproc(selfie_image)
        
            # if indent.selfie_msg is 'No face detected,please reupload selfie picture' or 'More than one face detected in the selfie, please upload only single face selfie image' or 'Selfie fails the Threshold check':
            #     raise Exception(indent.selfie_msg)
            if selfiescore is None:
                selfiescore = "NULL"
            #ap_temp_id = 1099 #static for now
            #ocr_decision = "Accepted" #static for now..
            
            ocr_error = ""
            
            if selfieresult == "PASS":
                #selfie test is success
                # ocr_error = "\""+"NULL"+"\""
                ocr_error ="NULL"
            
            elif selfieresult == "FAIL":
                #selfie test is Fail
                ocr_error = "\""+indent.selfie_msg+"\""
            
            # else:
            #     print("Invalid Selfie Result")
            #     print(selfieresult)
            #     response_json = {'result':selfieresult, 'ocr_error': indent.selfie_msg }
            
            dbcheckquery = "SELECT * FROM ocr_data WHERE ocr_ap_id_temp = {} ORDER BY ocr_id desc LIMIT 1".format(ap_temp_id)
            mycursor.execute(dbcheckquery)
            dbcheckres = mycursor.fetchall()
            if mycursor.rowcount > 0:
                #update query goes here
                updatequery = "UPDATE ocr_data SET ocr_selfie_score = "+str(selfiescore)+", ocr_selfie_result = "+"\""+selfieresult+"\""+", ocr_error = "+ocr_error+" WHERE ocr_ap_id_temp = "+str(ap_temp_id) + " ORDER BY ocr_id desc LIMIT 1"
                mycursor.execute(updatequery)
                cnx.commit()
                print("Row already exists updating")
        
            else:
                insertquery = "INSERT INTO ocr_data (ocr_ap_id_temp, ocr_selfie_score, ocr_selfie_result, ocr_error ,ocr_raw_response) VALUES (%s, %s ,%s, %s, %s)"
                ocr_raw_response = {'ocr_ap_id_temp': ap_temp_id, 'ocr_selfie_result':selfieresult,'ocr_error' : ocr_error}
                insertion_json = {'ocr_ap_id_temp': ap_temp_id, 'ocr_selfie_score':selfiescore, 'ocr_selfie_result':selfieresult,'ocr_error' : ocr_error,'ocr_raw_response' : str(ocr_raw_response)}
                val = tuple(insertion_json.values())
                mycursor.execute(insertquery, val)
                cnx.commit()
                print("Inserted into database")
            
            #phase 2 commented (if face matching needed hit another function)
            # indent.facematching(ktp_face_img,selfie_face_img)
            # indent.ktptextextract(cropped_img)
            #add columns
            
            #Closing the connection
            cnx.close()
            response_json = {'result':selfieresult, 'ocr_error': ocr_error}
            return retjson(response_json)
    
        except Exception as e:
            cnx.rollback()
            print("Exception caught Reverting SQL changes")
            selfieresult = "FAIL"
            print(e)
            #Closing the connection
            cnx.close()
            response_json = {'result':selfieresult, 'ocr_error': str(e) }
            return retjson(response_json)



# print(main('D:\\OCR-Final\\photos2\\30ktp.jpg',"D:/OCR-Final/photos2/30photo.jpg",'selfie'))
# print(main(cfg.debug_ktp_path,'ktp', cfg.debug_ap_id))
# print(main(cfg.debug_selfie_path,'selfie', cfg.debug_ap_id))


if __name__ == "__main__":
    app.run(host='0.0.0.0',port = '5000')
