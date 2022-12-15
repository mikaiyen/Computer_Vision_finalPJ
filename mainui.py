import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
import cvcaptcha as cp
from PIL import Image, ImageTk
from tkinter import messagebox
from tkinter import font as tkFont

def choosedata():
    global file_path
    file_path = filedialog.askopenfilename()   # 選擇檔案後回傳檔案路徑與名稱
    name_text.set(file_path)
    

def cvpredict():
    global img, tk_img
    if (name_text.get()!=""):
        usenewmodel=0
        if(cp.modelexist()):
            usetrainedmodel = messagebox.askyesno("trained model detected", "Detected trained model, do you want to use new model?")
            if(usetrainedmodel==True):
                usenewmodel=1
        Predict_text.set('Prediction:'+cp.predict(file_path,usenewmodel))
        img = Image.open(file_path)        # 開啟圖片
        img = img.resize((600,250))
        tk_img = ImageTk.PhotoImage(img)    # 轉換為 tk 圖片物件
        photolabel.config(image=tk_img)
        photolabel.pack()
    else:
        messagebox.showerror('Error', 'No data selected!')
    
    
def buildmodel():
    if batch_size_button.get()!="":
        bsize=int(batch_size_button.get())
    else:
        bsize=32
        
    if batch_size_button.get()!="":
        eps=int(epoch_button.get())
    else:
        eps=30
        
    if (cp.modelexist()):
        result = messagebox.askyesno("model detected", "Detected exist model, do you want to rebuild new model?")
        if(result==True):
            newmodel=cp.build_model(bsize, eps)
    else:
        newmodel=cp.build_model(bsize, eps)
    ifmodelsave = messagebox.askyesno("model built", "Do you want to save new model?")
    if(ifmodelsave==True):
        cp.savemodel(newmodel)
    
    
def overwritemodel():
    overwrite = messagebox.askyesno("Save model?", "Do you want to save the new model?")
    return overwrite




root = tk.Tk()
root.title('109502561')
root.geometry('800x500')
root.configure(bg='#97CBFF')

#-------------------------------
training_button = tk.Button(root, text='Training~（´◔​∀◔`)', command=buildmodel,fg="#2828FF",bg="#FFFFFB")
training_button.place(x=300,y=430)



predict_button = tk.Button(root, text='Predict~（´◔​∀◔`)', command=cvpredict,fg="#2828FF",bg="#FFFFFB")
predict_button.place(x=600,y=430)
#-------------------------------------------------
batch_size_button= tk.Entry(root)
batch_size_button.place(x=250, y=310)

epoch_button= tk.Entry(root)
epoch_button.place(x=250,y=350)

batch_size_label= tk.Label(root, text= 'batch size: (init=32)', fg="#2828FF",bg="#97CBFF").place(x=125, y=310)

epoch_label = tk.Label(root, text= "epoch: (init=30)", fg="#2828FF",bg="#97CBFF").place(x=125, y=350)

helv36 = tkFont.Font(family='Helvetica', size=20, weight='bold')
Predict_text=tk.StringVar()
Predict_text.set('Prediction:')
Predict_label = tk.Label(root, textvariable= Predict_text , fg="#2828FF",bg="#97CBFF")
Predict_label['font'] = helv36
Predict_label.place(x=450, y=310)

photolabel = tk.Label(root)  
#-------------------------------------------------
flower_label = tk.Label(root, text = "`*-.,.-*`♩`*-.,.-*`♪`*-.,.-*`♫`*-.,.-*`♬`*-.,.-*`♩`*-.,.-*`♪`*-.,.-*`♫`*-.,.-*`*-.,.-*`♩`*-.,.-*`♪`*-.,.-*`", fg="#2828FF",bg="#97CBFF",font=1)
flower_label.place(x=0, y=470)

#-----------------------------------------------------
choose_data = tk.Button(root,text='choose data',command=choosedata, fg="#2828FF",bg="#FFFFFB")
choose_data.place(x=10,y=310)
name_text = tk.StringVar()
name_label = tk.Label(root, textvariable=name_text, fg="#2828FF", bg="#97CBFF")
name_label.place(x=10, y=370)
#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

root.mainloop()