from firebase import firebase

firebase=firebase.FirebaseApplication("https://quickstart-1598859311837.firebaseio.com/",None)
data={
	'Name':'sandeep',
	'Email':'sandthapa46@gmail.com',
	'Phone':9841297261
}
result = firebase.post('/quickstart-1598859311837/student' , data)
print(result)
