import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, NumericProperty, ListProperty
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar

from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.providers.aer import AerSimulator

import math as m
import matplotlib.pyplot as plt
import numpy as np

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import time

Builder.load_file("1.kv")

#--------------------------------------------------------------------------------------
#FUNCIONES PARA CALCULAR EVOLUCIÓN TEMPORAL:
def qft(qc,n):
    '''performs quantum fourier transform on circuit qc
    n: number of qubits'''
    #iterates through each qubit starting at the bottom
    for j in range(n-1, -1, -1): 
    # h gate on each qubit    
        qc.h(j)
    #iterates through each qubit above current one
    #adds controlled P gate, phase pi/2**()
        for i in range(j-1, -1, -1):
            qc.cp(m.pi/2**(j-i), i,j)
    #creates barrier before next H gate
        qc.barrier()
    #changes order of qubits    
    for i in range(int(n/2)):
        qc.swap(i, n-i-1)

def inverse_qft(qc,n):
    '''performs inverse quantum fourier transform on circuit qc
    same as fourier, reversed order and phases are negative
    n: number of qubits'''
    #changes order of qubits    
    for i in range(int(n/2)):
        qc.swap(i, n-i-1)
    #iterates through each qubit starting at the bottom
    for j in range(0, n): 
    # h gate on each qubit    
        qc.h(j)
    #iterates through each qubit above current one
    #adds controlled P gate, phase -pi/2**()
        for i in range(j+1, n):
            qc.cp(-m.pi/2**(i-j), i,j)
    #creates barrier before next H gate
        qc.barrier()

def crz(qc, theta, control, target):
#controlled rz
# https://qiskit.org/textbook/ch-gates/more-circuit-identities.html
  qc.rz(theta/2,target)
  qc.cx(control,target)
  qc.rz(theta/2,target)
  qc.cx(control,target)
 

def tunnel(qc, t, n, h, well):
   '''
   Creates circuit qc needed to simulate schrodinger equation for 2 qubits

   qc: quantum circuit
   t: time increment
   n: number of time steps
   h: height of potential wells
   well: determines type of potential, if = 0, step potential, if = 1, double well
  
   '''
   for i in range(n):
     #fourier transf:
     qft(qc, 2)

     #kinetic operator  
     qc.rz(-t*m.pi**2,1)
     qc.rz(-t*0.25*m.pi**2,0)
     crz(qc,t*m.pi**2,0,1)

     #inverse fourier transform
     inverse_qft(qc, 2)

     #potential:
     #rotation around z axis
     if well == 0 or well == 1:
       qc.rz(2*h*t, 1-well)
     if well == 2:
       qc.rz(2*h*t, 0)
       qc.rz(2*h*t, 1)


    
   qc.measure_all()


def graphing(V_type, V_height):
    x_step = [0,1,2,2,3,4]
    y_step = [1,1,1,0,0,0]
    
    x_double = [0,1,1,2,3,3,4]
    y_double = [1,1,0,0,0,1,1] 
    
    x_single = [0,1,1,2,2,3,3,4]
    y_single = [1,1,0,0,1,1,0,0]
    
    font = {"size": 18}     
    plt.rc('font', **font)
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.set_ylim([0,1])
    ax1.set_ylabel("Probability $|\phi|^2$")
    ax1.set_xlim([0,4])
    
    ax2.set_ylabel("Potential V")
    ax2.set_ylim([0, 100])
    plt.xticks([i for i in range(4)],[" " for i in range(4)])   
    ax1.tick_params(axis='x',which='both',bottom=False, top=False,labelbottom=False)
    if V_type == 0:
        x = x_step
        y = [i*V_height for i in y_step]
        ax2.plot(x, y, "k--")
        ax2.fill_between(x, y, step="pre", alpha=0.4)    
        
    elif V_type == 1:
        x = x_single
        y = [i*V_height for i in y_single]
        ax2.plot(x, y, "k--")
        ax2.fill_between(x, y, step="pre", alpha=0.4)
        
    elif V_type == 2:
        x = x_double
        y = [i*V_height for i in y_double]
        ax2.plot(x, y, "k--")
        ax2.fill_between(x, y, step="pre", alpha=0.4)
    #plt.subplots_adjust(left=0.06, right=0.95, top=0.98, bottom=0.11)    

    
#--------------------------------------------------------------------------------------
#CREA GRAFICA INICIAL

graphing(0, 0)
class MyFigure(FigureCanvasKivyAgg):
    def __init__(self, **kwargs):
        super().__init__(plt.gcf(), **kwargs)

           
#--------------------------------------------------------------------------------------
class Game(Widget):
    #potential height (0-50)
    V_height = ObjectProperty(0) 

    #type of potential
    # 0 -> step
    # 1 -> single well
    # 2 -> double well
    V_type = ObjectProperty(0)
    
    # contains probability for each position and time 
    r = ListProperty()
    r = [[0] for i in range(4)]
    #counts time iteration
    counter = NumericProperty(0)
    
    def press(self, V):
        #sets type of potential, updates potential graph
        #tied to potential buttons
        ids_v = [self.step, self.double, self.single]
        
        #set type of potential
        self.V_type = V
        plt.close()
        graphing(self.V_type, self.V_height)
        
        #adds plot to layout
        canvas_plot=FigureCanvasKivyAgg(plt.gcf())
        self.axes.clear_widgets()
        self.axes.add_widget(canvas_plot)	
        
        #changes colors on potential buttons        
        for i in ids_v:
            i.background_color = [0.3, 0.3 , 0.3 ,1]
        ids_v[V].background_color = [1.0, 0.0, 0.0, 1.0]
        
        
        
        
    def V_slider(self, *args):
        #sets value for potential height
        #tied to potential slider
        self.V_height = args[1]
        self.potential_text.text = "Potential height: " + str(int(args[1]))
        plt.close()
        graphing(self.V_type, self.V_height)
        
        #adds plot to layout
        canvas_plot=FigureCanvasKivyAgg(plt.gcf())
        self.axes.clear_widgets()
        self.axes.add_widget(canvas_plot)
        
    def position_slider(self, *args, slider):
        #set initial state
        #tied to sliders
        ids_text = [self.text_00, self.text_01, self.text_10, self.text_11]
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        
        ids_text[slider].text = str(round(args[1], 2))
        if args[1] == 0:
            ids_sliders[slider].value_track = False
        else: 
            ids_sliders[slider].value_track = True
                
    def normalize(self):
        #normalizes position sliders, locks positions
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        norm = sum([i.value for i in ids_sliders])
        for i in ids_sliders:
            i.disabled = True
            if norm != 0:
                i.value = i.value/norm
            else: 
                i.value = 0.25
        self.calculate_button.disabled = False  
        self.initial_button.disabled = False    
        self.normalize_button.disabled = True  
                
            
    def release(self):
        # unlocks position sliders, potential buttons
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        self.step.disabled = False
        self.single.disabled = False
        self.double.disabled = False
        self.potential_slide.disabled = False
        
        self.normalize_button.disabled = False
        self.play_button.disabled = True
        self.initial_button.disabled = True
        self.big_graph.clear_widgets()
        
        for i in ids_sliders:
            i.disabled = False
    
    def calculate(self):
        #calculates probabilites for each position, time interval
        start = time.time()
    
        #self.loading.add_widget(ProgressBar())
        self.normalize_button.disabled = True
        self.initial_button.disabled = True
        self.calculate_button.disabled = True
        
        self.step.disabled = True
        self.single.disabled = True
        self.double.disabled = True
        self.potential_slide.disabled = True
        
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        qc = QuantumCircuit(2)
        sim = Aer.get_backend('aer_simulator') 
        delt = 0.1
        state = ["00", "01", "10", "11"]
        
        self.r = [[0 for i in range(50)] for i in range(4)]
        
        for i in range(50):
            qc = QuantumCircuit(2)
            qc.initialize([i.value**0.5 for i in ids_sliders])
            tunnel(qc, delt, i, self.V_height, self.V_type)
            result = sim.run(qc, shots=2**13).result()
            counts = result.get_counts()
            
            for j in range(len(state)):
                if state[j] in counts.keys():
                    self.r[j][i] = counts[state[j]]/(2**13)          
        self.play_button.disabled = False
        end = time.time()
        print(end-start)       
    def update(self, dt):
        #sets position sliders according to calculated probabilites
        
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        
        times = [0.1*i for i in range(50)]
        #stops updating sliders after 49 iterations:
        if self.counter > 49:
            self.play_button.disabled = False
            self.initial_button.disabled = False
            self.time.text = " "
            self.smol_graph.clear_widgets()
            canvas_plot2=FigureCanvasKivyAgg(plt.gcf())
            self.big_graph.add_widget(canvas_plot2)
            
            return False
            
        else: 
            self.time.text = "t = " + str(round(0.1*self.counter, 1))
            
            
            plt.close()
            fig2, ax3 = plt.subplots()
            ax3.set_xlim([0, 5])
            ax3.set_ylim([0,1])
            ax3.plot(times[:self.counter], self.r[0][:self.counter], "r-", label = "00")
            ax3.plot(times[:self.counter], self.r[1][:self.counter], "b-", label = "01")
            ax3.plot(times[:self.counter], self.r[2][:self.counter], "g-", label = "10")
            ax3.plot(times[:self.counter], self.r[3][:self.counter], "k-", label = "11")
            
            ax3.plot(times[self.counter-1], self.r[0][self.counter-1], "ro")
            ax3.plot(times[self.counter-1], self.r[1][self.counter-1], "bo")
            ax3.plot(times[self.counter-1], self.r[2][self.counter-1], "go")
            ax3.plot(times[self.counter-1], self.r[3][self.counter-1], "ko")
            
            ax3.set_ylabel("$|\phi|^2$")
            ax3.set_xlabel("Time t")
            
            ax3.legend()
            
            canvas_plot2=FigureCanvasKivyAgg(plt.gcf())
            self.smol_graph.clear_widgets()
            self.smol_graph.add_widget(canvas_plot2)
            
            for i in range(4):
                ids_sliders[i].value =  self.r[i][self.counter]
            self.counter += 1

    def play(self):
        #calls update function every 0.1 sec 
        self.big_graph.clear_widgets()
        self.counter = 0
        Clock.schedule_interval(self.update, 0.1)  
        self.play_button.disabled = True
        self.initial_button.disabled = True
        

    def spinner(self, value):
        self.big_graph.clear_widgets()
        self.play_button.disabled = False
        self.initial_button.disabled = True
        self.calculate_button.disabled = True
        self.normalize_button.disabled = True
        
        with open("demos.txt", "r") as file:
            a = file.readlines()
            data = [i.strip().split(",") for i in a]
            
            
        if value == "Double-Well":    
            self.r[0] = [float(i) for i in data[0]]
            self.r[1] = [float(i) for i in data[1]]
            self.r[2] = [float(i) for i in data[2]]
            self.r[3] = [float(i) for i in data[3]]
            
            self.slider_00.value = 0
            self.slider_01.value = 1
            self.slider_10.value = 0
            self.slider_11.value = 0
            
            self.step.disabled = True
            self.single.disabled = True
            self.double.disabled = False
            self.potential_slide.disabled = True
            
            self.potential_slide.value = 50
            
            self.V_type = 1
            plt.close()
            graphing(self.V_type, self.V_height)
        
            canvas_plot=FigureCanvasKivyAgg(plt.gcf())
            self.axes.clear_widgets()
            self.axes.add_widget(canvas_plot)	
            
            ids_v = [self.step, self.double, self.single]
            for i in ids_v:
                i.background_color = [0.3, 0.3 , 0.3 ,1]
            ids_v[1].background_color = [1.0, 0.0, 0.0, 1.0]
            
        if value == "Single-Well":    
            self.r[0] = [float(i) for i in data[5]]
            self.r[1] = [float(i) for i in data[6]]
            self.r[2] = [float(i) for i in data[7]]
            self.r[3] = [float(i) for i in data[8]]
            
            self.slider_00.value = 0
            self.slider_01.value = 0
            self.slider_10.value = 1
            self.slider_11.value = 0
            
            self.step.disabled = True
            self.single.disabled = False
            self.double.disabled = True
            self.potential_slide.disabled = True
            
            self.potential_slide.value = 80
            
            self.V_type = 2
            plt.close()
            graphing(self.V_type, self.V_height)
        
            canvas_plot=FigureCanvasKivyAgg(plt.gcf())
            self.axes.clear_widgets()
            self.axes.add_widget(canvas_plot)	
            
            ids_v = [self.step, self.double, self.single]
            for i in ids_v:
                i.background_color = [0.3, 0.3 , 0.3 ,1]
            ids_v[2].background_color = [1.0, 0.0, 0.0, 1.0]

        if value == "Step-Well":    
            self.r[0] = [float(i) for i in data[10]]
            self.r[1] = [float(i) for i in data[11]]
            self.r[2] = [float(i) for i in data[12]]
            self.r[3] = [float(i) for i in data[13]]
            
            self.slider_00.value = 0
            self.slider_01.value = 0
            self.slider_10.value = 0
            self.slider_11.value = 1
            
            self.step.disabled = False
            self.single.disabled = True
            self.double.disabled = True
            self.potential_slide.disabled = True
            
            self.potential_slide.value = 50
            
            self.V_type = 0
            plt.close()
            graphing(self.V_type, self.V_height)
        
            canvas_plot=FigureCanvasKivyAgg(plt.gcf())
            self.axes.clear_widgets()
            self.axes.add_widget(canvas_plot)	
            
            ids_v = [self.step, self.double, self.single]
            for i in ids_v:
                i.background_color = [0.3, 0.3 , 0.3 ,1]
            ids_v[0].background_color = [1.0, 0.0, 0.0, 1.0]
                    
        
        if value == "Free Particle":    
            self.r[0] = [float(i) for i in data[15]]
            self.r[1] = [float(i) for i in data[16]]
            self.r[2] = [float(i) for i in data[17]]
            self.r[3] = [float(i) for i in data[18]]
            
            self.slider_00.value = 1
            self.slider_01.value = 0
            self.slider_10.value = 0
            self.slider_11.value = 0
            
            self.step.disabled = True
            self.single.disabled = True
            self.double.disabled = True
            self.potential_slide.disabled = True
            
            self.potential_slide.value = 0
            
            self.V_type = 0
            plt.close()
            graphing(self.V_type, self.V_height)
        
            canvas_plot=FigureCanvasKivyAgg(plt.gcf())
            self.axes.clear_widgets()
            self.axes.add_widget(canvas_plot)	
            
        
        
        
                    
class sliide(App):
    def build(self):
        return Game()
    
if __name__ == "__main__":
    sliide().run()
    
#1.8971142768859863
#1.973564863204956
#2.2491369247436523    