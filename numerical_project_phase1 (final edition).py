#!/usr/bin/env python
# coding: utf-8

# # Numerical project phase 1

# In[24]:


import tkinter as tk
from tkinter.ttk import *
import numpy as np
import pandas as pd
import math
from tkinter import font
from timeit import default_timer as timer
from tkinter import messagebox 


# Gauss Elimination and Gauss Jordan

# In[25]:


class Basics :
    def __init__ (self, AB, fig):
        AB = np.split(AB, (len(AB), len(AB[0])), 1)
        A = AB[0]
        Bspare = AB[1]
        B = []
        for i in range(len(Bspare)):
            B.append(Bspare[i][0])
        self.A = A
        self.B = B
        self.ln = len(B)
        self.fig = fig
        
    def precision(self, num):
        if(num == 0):
            return num
        result = round(num, -int(math.floor(math.log10(abs(num)))) + (self.fig-1))
        if (result is None):
            return num
        return result
    
    def Pivot_without_scaling(self, k):
        p = k
        
        #Detecting the largest scaled number (pivot) in column k
        big = abs(self.A[k][k])
        big = Basics.precision(self, big)
        for i in range(k+1, self.ln):
            dummy = abs(self.A[i][k])
            dummy = Basics.precision(self, dummy)
            if dummy > big :
                big = dummy
                p = i
        
        #If a new pivot is detected, swap the rows
        if p != k :
            for j in range(k , self.ln):
                dummy = self.A[p][j]
                self.A [p][j] = self.A[k][j]
                self.A[k][j] = dummy
            
            #Swapping the values in B
            dummy = self.B[p]
            self.B[p] = self.B[k]
            self.B[k] = dummy
            
        return self.A, self.B
    
    def Pivot_with_scaling (self, S, k):
        p = k
        
        #Detecting the largest scaled number (pivot) in column k
        big = abs(self.A[k][k] / S[k])
        big = Basics.precision(self, big)
        for i in range(k+1, self.ln):
            dummy = abs(self.A[i][k] / S[i])
            dummy = Basics.precision(self, dummy)
            if dummy > big :
                big = dummy
                p = i
        
        #If a new pivot is detected, swap the rows
        if p != k :
            for j in range(k , self.ln):
                dummy = self.A[p][j]
                self.A [p][j] = self.A[k][j]
                self.A[k][j] = dummy
            
            #Swapping the values in B
            dummy = self.B[p]
            self.B[p] = self.B[k]
            self.B[k] = dummy
            
            #Swapping the values in S
            dummy = S[p]
            S[p] = S[k]
            S[k] = dummy
            
        return self.A, self.B, S
    
    def ForwardElimination_without_scaling (self):
        for k in range(self.ln - 1):
            self.A, self.B = Basics.Pivot_without_scaling(self, k)
            for i in range(k+1, self.ln):
                factor = self.A[i][k] / self.A[k][k]
                factor = Basics.precision(self, factor)
                for j in range(k+1, self.ln):
                    self.A[i][j] = self.A[i][j] - factor * self.A[k][j]
                    self.A[i][j] = Basics.precision(self, self.A[i][j])
                self.B[i] = self.B[i] - factor * self.B[k]
                self.B[i] = Basics.precision(self, self.B[i])
        return self.A, self.B
    
    def ForwardElimination_with_scaling (self, S):
        for k in range(self.ln - 1):
            self.A, self.B, S = Basics.Pivot_with_scaling(self, S, k)
            for i in range(k+1, self.ln):
                factor = self.A[i][k] / self.A[k][k]
                factor = Basics.precision(self, factor)
                for j in range(k+1, self.ln):
                    self.A[i][j] = self.A[i][j] - factor * self.A[k][j]
                    self.A[i][j] = Basics.precision(self, self.A[i][j])
                self.B[i] = self.B[i] - factor * self.B[k]
                self.B[i] = Basics.precision(self, self.B[i])
        return self.A, self.B
    
    def BackSubstitution (self):
        X = [None] * self.ln
        X[self.ln-1] = self.B[self.ln-1] / self.A[self.ln-1][self.ln-1]
        X[self.ln-1] = Basics.precision(self, X[self.ln-1])
        for i in range(self.ln-1, -1, -1):
            sum = 0
            for j in range(i+1, self.ln):
                sum = sum + self.A[i][j] * X[j]
                sum = Basics.precision(self, sum)
            X[i] = (self.B[i] - sum) / self.A[i][i]
            X[i] = Basics.precision(self, X[i])
        return self.A, self.B, X
        
    def BackElimination (self):
        for k in range(self.ln-1, -1, -1):
            for i in range(k-1, -1, -1):
                factor = self.A[i][k] / self.A[k][k]
                factor = Basics.precision(self, factor)
                self.B[i] = self.B[i] - factor * self.B[k]
                self.B[i] = Basics.precision(self, self.B[i])
        return self.A, self.B
    
    def printX(self, array):
        tmp = ""
        for i in range (len(array)):
            tmp += "X" + str(i+1) + " = "  + str(array[i]) + "\n"
        return tmp
        


# In[26]:


class GaussJordan (Basics):
    
    def solve_without_scaling(self, basic):
        rankA = np.linalg.matrix_rank(self.A)
        AB = np.column_stack([self.A, self.B])
        rankAaugB = np.linalg.matrix_rank(AB)
        if rankA == rankAaugB and rankA == self.ln :
            self.A, self.B = basic.ForwardElimination_without_scaling()
            self.A, self.B = basic.BackElimination()
            for k in range(self.ln):
                self.B[k] = self.B[k] / self.A[k][k]
                self.B[k] = Basics.precision(self, self.B[k])
            out = basic.printX(self.B)
            
        elif rankA != rankAaugB:
            out = "This linear system of equations has no solution."
       
        else:
            out = "This linear system of equations has infinite number of solutions."
        
        return out
    
    def solve_with_scaling(self, basic):
        rankA = np.linalg.matrix_rank(self.A)
        AB = np.column_stack([self.A, self.B])
        rankAaugB = np.linalg.matrix_rank(AB)
        if rankA == rankAaugB and rankA == self.ln :
            S = [None] * self.ln
            for i in range(self.ln):
                S[i] = abs(self.A[i][0])
                for j in range(1, self.ln):
                    S[i] = max(abs(self.A[i][j]), S[i])
        
            self.A, self.B = basic.ForwardElimination_with_scaling(S)
            self.A, self.B = basic.BackElimination()
            for k in range(self.ln):
                self.B[k] = self.B[k] / self.A[k][k]
                self.B[k] = Basics.precision(self, self.B[k])
            out = basic.printX(self.B)
            
        elif rankA != rankAaugB:
            out = "This linear system of equations has no solution."
       
        else:
            out = "This linear system of equations has infinite number of solutions."
        
        return out
    


# In[27]:


class GaussElimination (Basics):
    
    def solve_without_scaling (self, basic):
        rankA = np.linalg.matrix_rank(self.A)
        AaugB = np.column_stack([self.A, self.B])
        rankAaugB = np.linalg.matrix_rank(AaugB)
        if rankA == rankAaugB and rankA == self.ln :
            self.A, self.B = basic.ForwardElimination_without_scaling()
            self.A, self.B, X = basic.BackSubstitution()
            out = basic.printX(X)
            
        elif rankA != rankAaugB:
            out = "This linear system of equations has no solution."
        
        else:
            out = "This linear system of equations has infinite number of solutions."
        
        return out
    
    def solve_with_scaling (self, basic):
        rankA = np.linalg.matrix_rank(self.A)
        AaugB = np.column_stack([self.A, self.B])
        rankAaugB = np.linalg.matrix_rank(AaugB)
        if rankA == rankAaugB and rankA == self.ln :
            S = [None] * self.ln
            for i in range(self.ln):
                S[i] = abs(self.A[i][0])
                for j in range(1, self.ln):
                    S[i] = max(abs(self.A[i][j]), S[i])
        
            self.A, self.B = basic.ForwardElimination_with_scaling(S)
            self.A, self.B, X = basic.BackSubstitution()
            out = basic.printX(X)
            
        elif rankA != rankAaugB:
            out = "This linear system of equations has no solution."
        
        else:
            out = "This linear system of equations has infinite number of solutions."
        
        return out
        


# LU Decomposition

# In[28]:



from pickle import NONE
from abc import abstractclassmethod
class Tools:
    def precision(self, num, p):
        if(abs(num) < 1e-10 or pd.isna(num)):
            return num
        result = round(num, -int(math.floor(math.log10(abs(num)))) + (p-1))
        if (result is None):
            return num
        return result
    def checkSolution(self,A,B):
        rankA = np.linalg.matrix_rank(A)
        AB = np.column_stack([A, B])
        rankAB = np.linalg.matrix_rank(AB)
        if rankA == rankAB and rankA == len(A):
            out = "has a unique solution"
        elif rankA != rankAB:
            out = "This linear system of equations has no solution."

        else:
            out = "This linear system of equations has infinite number of solutions."
        return out


# In[29]:


class LUDecomposition:
    def __init__(self, A, fig):
        self.A = A
        self.n = len(A)
        self.L = np.zeros((self.n, self.n), dtype=float)
        self.U = np.zeros((self.n, self.n), dtype=float)
        self.y = None
        self.solution = None
        self.fig = fig
    @abstractclassmethod
    def decompose(self):
        pass

    def forward_substitution(self, b):
        fig = Tools()
        y = np.zeros(self.n)
        for i in range(self.n):
            y[i] = fig.precision((b[i] - np.sum(self.L[i, :i] * y[:i])) / self.L[i, i],self.fig)
        return y

    def backward_substitution(self):
        fig = Tools()
        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            x[i] = fig.precision((self.y[i] - np.sum(self.U[i, i+1:] * x[i+1:])) / self.U[i, i],self.fig)
        return x

    def solve_linear_system(self,P,b):
            fig = Tools()
            fig.checkSolution(self.A,b)
            Pb = self.P @ b.T
            if(fig.checkSolution(self.A,b) == "has a unique solution" ):
                self.y = self.forward_substitution(Pb)
                self.solution = self.backward_substitution()
                return self.get_results()
            else: return fig.checkSolution(self.A,b)

    def get_results(self):
        result_str = ""
        for i in range (self.n):
            result_str += "X" + str(i+1) + " = "  + str(self.solution[i]) + "\n"
        return result_str


# In[30]:


class DoolittleLU(LUDecomposition):
    def __init__(self, A, fig):
        self.A = A
        self.n = len(A)
        self.P = np.eye(self.n)
        self.L = np.eye(self.n)
        self.U = A.astype(float)
        self.fig = fig

    def decompose(self,b):
        fig = Tools()

        for k in range(self.n - 1):
            pivot = self.U[k, k]

            if np.abs(pivot) < 1e-10:
                # Find the index of the first nonzero pivot in the current column below the current row
                nonzero_row = np.argmax(np.abs(self.U[k:, k]) >= 1e-10) + k             
                self.U[[k, nonzero_row], :] = self.U[[nonzero_row, k], :]
                self.L[[k, nonzero_row], :k] = self.L[[nonzero_row, k], :k]
                self.P[[k, nonzero_row], :] = self.P[[nonzero_row, k], :]

                pivot = self.U[k, k]  # Update pivot after pivoting

            for i in range(k + 1, self.n):
                factor = fig.precision(self.U[i, k] / pivot,self.fig)
                self.L[i, k] = factor
                self.U[i, k:] -= factor * self.U[k, k:]
                for j in range(k,self.n):
                    self.U[i, j] = fig.precision(self.U[i, j],self.fig)
        return super().solve_linear_system(self.P,b)


# In[31]:


class CholeskyLU(LUDecomposition):
    def __init__(self, A, fig):
        self.A = A
        self.n = len(A)
        self.L = np.zeros_like(A, dtype=float)
        self.valid = True
        self.U = np.zeros_like(A, dtype=float)
        self.fig = fig
        self.P = np.eye(self.n)
        self.err_msg=""



    def check_solvability(self):
        if not self.is_symmetric_positive_definite(self.A):
            self.err_msg=("Matrix must be symmetric positive definite for Cholesky decomposition.")
            self.valid = False
            return self.err_msg



    def is_symmetric_positive_definite(self, A):
        return np.allclose(A, A.T) and np.all(np.linalg.eigvals(A) > 0)

    def cholesky_decomposition(self,b):
        self.check_solvability()
        fig = Tools()
        if(self.valid):
            for i in range(self.n):
                for j in range(i + 1):
                    if i == j:
                        sum_val = fig.precision(sum(self.L[i, k] ** 2 for k in range(j)),self.fig)
                        self.L[i, i] = fig.precision(np.sqrt(self.A[i, i] - sum_val),self.fig)
                    else:
                        sum_val = fig.precision(sum(self.L[i, k] * self.L[j, k] for k in range(j)),self.fig)
                        self.L[i, j] = fig.precision((self.A[i, j] - sum_val) / self.L[j, j],self.fig)


            self.U = self.L.T
            return super().solve_linear_system(self.P,b)
        else: return self.check_solvability()


# In[32]:


class CroutLU(LUDecomposition):
    def __init__(self, A,fig):
        self.A = A
        self.n = len(A)
        self.L = np.zeros((self.n, self.n))
        self.U = np.eye(self.n)
        self.isFailed = False
        self.fig = fig
        self.err_msg=""
        self.P = np.eye(self.n)

    def Crout_decompose(self,b):
        fig = Tools()
        for j in range(self.n):
            for i in range(j, self.n):
                self.L[i, j] = fig.precision(self.A[i, j] - np.sum(self.L[i, :j] * self.U[:j, j]),self.fig)

                if np.abs(self.L[j, j]) < 1e-10:
                    self.err_msg="Division by zero encountered. Crout method failed."
                    self.isFailed = True
                    
                else:
                    self.U[j, i] = fig.precision((self.A[j, i] - np.sum(self.L[j, :j] * self.U[:j, i])) / self.L[j, j],self.fig)
        if(self.isFailed == False):
            return super().solve_linear_system(self.P,b)
        else: return (self.err_msg)



# In[33]:


class LUFormat:
    def __init__(self, matrix, fig,b):
        self.matrix = matrix
        self.fig = fig
        self.b = b

    def format_and_print(self, method_name):
        method_name = method_name.lower()

        if method_name == "doolittle form":
            return self.doolittle_format(self.b)
        elif method_name == "cholesky form":
            return self.cholesky_format(self.b)
        elif method_name == "crout form":
            return self.crout_format(self.b)
        else:
            print(f"Unsupported LU decomposition method: {method_name}")

    def doolittle_format(self,b):
        doolittle_solver = DoolittleLU(self.matrix,self.fig)
        return doolittle_solver.decompose(b)


    def cholesky_format(self,b):
        cholesky_solver = CholeskyLU(self.matrix,self.fig)
        return cholesky_solver.cholesky_decomposition(b)


    def crout_format(self,b):
        crout_solver = CroutLU(self.matrix,self.fig)
        return crout_solver.Crout_decompose(b)


# Gauss Sedel and Jacobi

# In[34]:


class tools:
    def printX(self, array):
        tmp = ""
        for i in range (len(array)):
            tmp += "x" + str(i+1) + " = "  + str(array[i]) +"\n"
        return tmp
    
    def precision(self, num, p):
        if(num == 0):
            return num
        result = round(num, -int(math.floor(math.log10(abs(num)))) + (p-1))
        if (result is None):
            return num
        return result

    def is_correct(self, array):
        for i in range(len(array)-1):
            for j in range(len(array)-1):
                if(i == j):
                    if(array[i][i] == 0):
                        return False
        return True  


# In[35]:


class Gauss_Seidel:
    def seidel(self, array, initial, iteration, error, p):
        tool = tools()
        if(tool.is_correct(array)):
            iterat = 0
            while(iterat < iteration):
                condition = False
                E = []
                for i in range(len(array)):
                    x = tool.precision(array[i][-1], p)
                    for j in range (len(array[0])-1):
                        if(j != i):
                            x -= tool.precision(array[i][j], p)*initial[j]
                    x /= array[i][i]
                    if(x != 0):
                        E.append(tool.precision(abs((x - initial[i])/x)*100,p))
                    else:
                        E.append(math.inf)
                    initial[i] = tool.precision(x, p)
                for k in range(len(array)):
                    condition = condition or E[k]>error
                if(not condition):
                    return tool.printX(initial)
                iterat += 1
            return "system is not converging"
        else:
            return "can't solve system"


# In[45]:


class Jacobi:
    def jacobi(self, array, initial, iteration, error, p):
        tool = tools()
        if(tool.is_correct(array)):
            iterat = 0
            while(iterat < iteration):
                temp=[]
                E = []
                condition = False
                for i in range(len(array)):
                    x = tool.precision(array[i][-1],p)
                    for j in range (len(array[0])-1):
                        if(j != i):
                            x -= tool.precision(array[i][j], p)*initial[j]
                    x /= array[i][i]
                    temp.append(tool.precision(x,p))
                    if(x != 0):
                        E.append(tool.precision(abs((x - initial[i])/x)*100,p))
                    else:
                        E.append(math.inf)
                for k in range(len(array)):
                    initial[k] = temp[k]
                for k in range(len(array)):
                    condition = condition or E[k]>error
                if(not condition):
                    return tool.printX(initial)
                iterat += 1
            return "system is not converging"
        else:
            return "can't solve system"


#Example
array = [[5,2,4], 
    [3,6,5],
    ]
initial = [0,0]

solve =Jacobi()
print(solve.jacobi(array,initial,50,0.1,5))


# GUI 

# In[46]:


#GUI part
#display the grid using number of equations
def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def check(s):
    if not(is_float(s)):
        messagebox.showerror("showerror", "Invalid Input \nPlease enter numbers") 
        return False
    else:
        return True
    
def entries(k):
    equ.clear()
    for widget in frame.winfo_children():
        widget.destroy()
        
    for i in range (k):
        equ.append([])
        for j in range (k*2):
            if(j%2==0):
                entry = Entry(master=frame , width=10)
                entry.grid(row =i , column=j)
                entry.insert(0,0)
                equ[i].append(entry)
            else:
                if(j != k*2-1):
                    label = Label(master=frame, text=f"X{math.floor((j+1)/2)} + ")
                
                else :
                    label = Label(master=frame, text=f"X{int((j+1)/2)} = ")
                label.grid(row=i , column=j)
            
        ent_eq =Entry(master=frame, width=10)
        ent_eq.grid(row=i,column= j+1)
        ent_eq.insert(0,0)
        equ[i].append(ent_eq)

#taking number of equations from the user
def selectednumber(event):
    k= clicked.get()
    entries(k)
        
    if clickedmethod.get()=="Gauss Sediel" or clickedmethod.get()=="Jacobi Iteration":
        selectedmethod(event)
        
def inti_sel_method ():
    var = tk.StringVar(value='A')
    r1 =Radiobutton(frm_parameters, text='with scaling', variable=var, value='A', command=None)
    r1.grid(row=1,column=1)
    r2 =Radiobutton(frm_parameters, text='without scaling', variable=var, value='B'  ,command=None)
    r2.grid(row=1,column=2)
    parameters.append(var)
def selectedmethod(event):
    for widget in frm_parameters.winfo_children():
        widget.destroy()
    parameters.clear()
    dropdown_label=Label(frm_parameters,text="the format of L & U")
    dropdown_option=tk.StringVar(frm_parameters)
    
    if clickedmethod.get() == "LU Decomposition":
        format_options=("Doolittle Form","Crout Form", "Cholesky Form")
        format_list =OptionMenu(frm_parameters,dropdown_option,format_options[0],*format_options)
        dropdown_label.grid(row=1,column=0)
        format_list.grid(row=1,column=1)
        parameters.append(dropdown_option)
    elif clickedmethod.get() == "Gauss Elimination" or clickedmethod.get() == "Gauss Jordan":
        inti_sel_method()
    elif clickedmethod.get()=="Gauss Sediel" or clickedmethod.get()=="Jacobi Iteration" :
        guess_label=Label(frm_parameters,text="The initial guess :")
        guess_label.grid(row=1,column=0)
        parameters.append([])
        for i in range (clicked.get()):
            var_label=Label(frm_parameters,text=f"X{i+1}").grid(row=i+2,column=0)
            ent = Entry(frm_parameters , width=10)
            ent.insert(0,0)
            ent.grid(row=i+2,column=1)
            parameters[0].append(ent)
        dropdown_label['text']="The Stopping Conditions :"
        dropdown_label.grid(row=1,column=3)
        num_iter_label=Label(frm_parameters,text='MAX Number of Iterations').grid(row=2,column=3)
        num_iter_entry=Entry(frm_parameters , width=10)
        num_iter_entry.grid(row=2,column=4)
        parameters.append(num_iter_entry)
        error_label=Label(frm_parameters,text='Absolute Relative Error').grid(row=3,column=3)
        error_entry=Entry(frm_parameters , width=10)
        error_entry.grid(row=3,column=4)
        error_entry.grid(row=3,column=4)
        parameters.append(error_entry)
        Separator(frm_parameters, orient=tk.VERTICAL).grid(column=2, row=1, rowspan=len(parameters[0])+1, sticky='ns',padx=10)

#store the values that user entered to start solving the system
def solve(): 
    equ_arr = []
    solution='Solution : \n\n'
    if default.get().isdigit():
        precision=int(default.get())
        if(precision > 10 or precision <1):
            messagebox.showerror("showerror", "Invalid Input \nThe range of precision 1:10") 
            return
    else:
        messagebox.showerror("showerror", "Invalid Input \nPrecision must be Positive Integer") 
        return
    start=0
    end=0
    for i in range (len(equ)):
        equ_arr.append([])
        for j in range (len(equ[i])):
            if check(equ[i][j].get()):
                equ_arr[i].append(float(equ[i][j].get()))
            else :return
            
    if clickedmethod.get() == "Gauss Elimination":
        #solve using gauss elimination here
        start = timer()
        b = Basics(np.array(equ_arr), precision)
        pe = GaussElimination(np.array(equ_arr), precision)
        if(parameters[0].get() == 'A'):
            solution += pe.solve_with_scaling(b)
        else:
            solution += pe.solve_without_scaling(b)
        end = timer()
        
    elif clickedmethod.get() == "Gauss Jordan":
        start = timer()
        b = Basics(np.array(equ_arr), precision)
        pe = GaussJordan(np.array(equ_arr), precision)
        #solve using gauss jordan here
        if(parameters[0].get() == 'A'):
            solution += pe.solve_with_scaling(b)
        else:
            solution += pe.solve_without_scaling(b)
        end = timer()
        
    elif clickedmethod.get() == "LU Decomposition":
        b= np.array([row[-1] for row in equ_arr])
        A = np.delete(equ_arr,np.s_[-1:],axis = 1)
        sel_method = parameters[0].get()
        start = timer()
        LU_Solve = LUFormat(np.array(A),precision,b)
        if(LU_Solve.format_and_print(sel_method) is not None):
            solution+=LU_Solve.format_and_print(sel_method)
            end = timer()
        #solve using lu decomposition here
    else :
        inti_guess=[]
        for i in range (len(parameters[0])):
            if check(parameters[0][i].get()):
                inti_guess.append(float(parameters[0][i].get()))
            else :return
    
        if not (parameters[1].get()).isdigit() :
            messagebox.showerror("showerror", "Invalid Input \nNumber of iterations must be Positive Integer") 
            return
        if not check(parameters[2].get()):
            return
        if float(parameters[2].get()) < 0:
            messagebox.showerror("showerror", "Invalid Input \nRelative error must be Positive number") 
            return
        
        iter_num=int(parameters[1].get())
        error_value=float(parameters[2].get())
        if clickedmethod.get() == "Gauss Sediel":
            start = timer()
            #solve using gauss sediel
            solve=Gauss_Seidel()
            solution+=solve.seidel(equ_arr,inti_guess,iter_num,error_value,precision)
            end = timer()
        elif clickedmethod.get() == "Jacobi Iteration":
            start = timer()
            #solve using jacobi
            solve =Jacobi()
            solution+=solve.jacobi(equ_arr,inti_guess,iter_num,error_value,precision)
            end = timer()
    dis_sol['text']=solution  
    dis_time['text']=f'Run Time :{end-start}'
                
equ=[]
parameters=[]
precision=5
window = tk.Tk()
window.geometry("600x600")
window.title("Linear equations solver")
window.option_add("*Font", "aerial 10")
window['pady'] = 25
#display the number of equations 
frm_num_equ = Frame(window)
frm_num_equ.pack()

num_equ_label=Label(frm_num_equ,text="Number of equations in the system:").grid(row=0, column=0)
options=[2,3,4,5,6,7,8]
clicked = tk.IntVar()
num_equ_chosen=OptionMenu(frm_num_equ, clicked, options[0], *options, command=selectednumber)
num_equ_chosen.grid(row=0, column=1)
frame = Frame()
frame.pack(pady=10)

#display the intial grid to input the equations
entries(2)

#display the precision 
frm_p=Frame()
frm_p.pack(pady=10)
label_p=Label(frm_p,text="Number of significant bits:").grid(row=0,column=0)
default = tk.StringVar(value=5)
p = Spinbox(frm_p, from_=1, to=10, textvariable=default, width =8).grid(row=0,column=1)

#display the methods that used to solve the system
frm_methods = Frame()
frm_methods.pack(pady=10)
frm_parameters = Frame()
frm_parameters.pack(pady=10)
label_method=Label(frm_methods,text="Solving by: ").grid(row=0,column=0)
methods=["Gauss Elimination","Gauss Jordan","LU Decomposition","Gauss Sediel","Jacobi Iteration"]
clickedmethod=tk.StringVar(frm_methods)
method_chosen=OptionMenu(frm_methods,clickedmethod,methods[0],*methods,command=selectedmethod)
method_chosen.grid(row=0, column=1)
inti_sel_method()
solve_btn = Button(text="solve" ,command=solve)
solve_btn.pack(pady=10)
dis_sol=Label(text='')
dis_sol.pack(pady=20)
dis_time=Label(text='')
dis_time.pack(pady=10,padx=10,anchor="se")
window.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




