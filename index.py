import pandas as pd
import numpy as np

def convertRange(x):
    temp = x.split('_')
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None

def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        
        st = np.std(subdf.price_per_sqft)
        
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output,gen_df],ignore_index =True)
    return df_output

def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
              bhk_stats['bhk'] = {
                  'mean': np.mean(bhk_df.price_per_sqft),
                  'std': np.std(bhk_df.price_per_sqft),
                  'count': bhk_df.shape[0]
              }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


data=pd.read_csv('Bengaluru_House_Data.csv')
data.drop(columns=['area_type','availability','society','balcony'],inplace=True)
data['location'] = data['location'].fillna('Sarjapur Road')
data['size'] = data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())
data['bhk']=data['size'].str.get(0).astype(int)
data['total_sqft']=data['total_sqft'].apply(convertRange)
data['price_per_sqft'] = data['price'] *100000 / data['total_sqft']
data['location']=data['location'].apply(lambda x: x.strip())
location_count= data['location'].value_counts()
location_count_less_10 = location_count[location_count<=10]
location_count_less_10
data['location']=data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)
data = data[((data['total_sqft']/data['bhk']) >= 300)]
data = remove_outliers_sqft(data)
data = bhk_outlier_remover(data)
data.drop(columns=['size','price_per_sqft'],inplace=True)
# X=data.drop(columns=['location','price'])
X=data.drop(columns=['price'])
y=data['price']
print('Shape of X = ', X.shape)
print('Shape of y = ', y.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)
column_trans = make_column_transformer((OneHotEncoder(sparse=False), ['location']),remainder='passthrough')
scaler = StandardScaler()
lr = LinearRegression(normalize=True)
pipe=make_pipeline(column_trans,scaler, lr)
pipe.fit(X_train,y_train)






from fnmatch import fnmatch
from tkinter import* 
from tkinter import ttk
import tkinter
from PIL import Image,ImageTk
from tkinter import messagebox

root=Tk()
root.title("Register")
root.geometry("1800x1000+0+0")

areaType=StringVar()
location=StringVar()
status=StringVar()
size=IntVar()
society=StringVar()
total_sqft=IntVar()
bath=IntVar()
balcony=IntVar()
price=IntVar()

#Background Image
bg=ImageTk.PhotoImage(file="./Background.jpg") #Background 
lbl_bg=Label(root,image=bg) #window space
lbl_bg.place(x=0,y=0,relwidth=1,relheight=1)



Title_Label=Label(root,text="Bengaluru House Price Prediction",font=("times new roman",25,"bold"),bg="white smoke")
Title_Label.place(x=580,y=35)


areaType_Label=Label(root,text="House Area Type",font=("times new roman",15,"bold"),bg="white")
areaType_Label.place(x=540,y=100,width=250)
areaType_entry=ttk.Entry(root,textvariable=areaType,font=("times new roman",15))
areaType_entry.place(x=860,y=100,width=250)


location_Label=Label(root,text="House Location",font=("times new roman",15,"bold"),bg="white")
location_Label.place(x=540,y=145,width=250)
location_entry=ttk.Entry(root,textvariable=location,font=("times new roman",15))
location_entry.place(x=860,y=145,width=250)


status_Label=Label(root,text="Moving in Status",font=("times new roman",15,"bold"),bg="white")
status_entry=ttk.Entry(root,textvariable=status,font=("times new roman",15))
status_Label.place(x=540,y=190,width=250)
status_entry.place(x=860,y=190,width=250)


size_Label=Label(root,text="Size of House(in BHK)",font=("times new roman",15,"bold"),bg="white")
size_entry=ttk.Entry(root,textvariable=size,font=("times new roman",15))
size_Label.place(x=540,y=235,width=250)
size_entry.place(x=860,y=235,width=250)

society_Label=Label(root,text="Society Name",font=("times new roman",15,"bold"),bg="white")
society_entry=ttk.Entry(root,textvariable=society,font=("times new roman",15))
society_entry.place(x=860,y=280,width=250)
society_Label.place(x=540,y=280,width=250)


total_sqft_Label=Label(root,text="Total area(sqft)",font=("times new roman",15,"bold"),bg="white")
total_sqft_entry=ttk.Entry(root,textvariable=total_sqft,font=("times new roman",15))
total_sqft_Label.place(x=540,y=325,width=250)
total_sqft_entry.place(x=860,y=325,width=250)


bath_Label=Label(root,text="Toilet/Bath No.",font=("times new roman",15,"bold"),bg="white")
bath_entry=ttk.Entry(root,textvariable=bath,font=("times new roman",15))
bath_Label.place(x=540,y=370,width=250)
bath_entry.place(x=860,y=370,width=250)


balcony_Label=Label(root,text="Balcony No.",font=("times new roman",15,"bold"),bg="white")
balcony_entry=ttk.Entry(root,textvariable=balcony,font=("times new roman",15))
balcony_Label.place(x=540,y=415,width=250)
balcony_entry.place(x=860,y=415,width=250)

def myClick():
    try:
        if location.get()=="" or total_sqft.get()=="" or size.get()=="" or bath.get()=="" :
            messagebox.showerror("Error","All feilds are required !!!",parent=root)
        else:
            messagebox.showinfo("Success","Entered successfully",parent=root)
            NewIn=[[location.get(),total_sqft.get(),bath.get(),size.get()]]
            NewIn=pd.DataFrame(NewIn)
            NewIn.columns=['location','total_sqft','bath','bhk']
            print(NewIn)
            try:
                y_pred_lr = pipe.predict(NewIn)
                y_pred_lr=str(int(y_pred_lr))+"0"+" Rupees/sqft"
                # print(y_pred_lr)
                price.config(text=y_pred_lr)
            except:
                messagebox.showinfo("Error","No Prediction made. Enter correct values!!!",parent=root)
    except:
        messagebox.showinfo("Error","Enter values of correct datatype!!!",parent=root)

        
        

myButton=Button(root,text="Predict Price",font=("times new roman",20,"bold"),bg="light goldenrod",command=myClick)
myButton.place(x=540,y=490,width=570)

price_Label=Label(root,text="Price",font=("times new roman",15,"bold"),bg="white")
price_Label.place(x=540,y=585,width=250)
price=Label(root,text="",font=("times new roman",15,"bold"),bg="white")
price.place(x=860,y=585,width=250)



root.mainloop()