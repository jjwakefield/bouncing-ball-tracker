import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore


class MainWindow(QtWidgets.QMainWindow):

     def __init__(self, actual, measurements, filtered, x_range, y_range, update_interval):
          super(MainWindow, self).__init__()

          self.graphWidget = pg.PlotWidget()
          self.graphWidget.setXRange(x_range[0], x_range[1])
          self.graphWidget.setYRange(y_range[0], y_range[1])
          self.setCentralWidget(self.graphWidget)

          self.actual = actual
          self.measurements = measurements
          self.filtered = filtered

          self.actual_plot = np.array([[actual[0, 0], actual[0, 1]]])
          self.measurement_plot = np.array([[measurements[0, 0], measurements[0, 1]]])
          self.filtered_plot = np.array([[filtered[0, 0], filtered[0, 1]]])

          self.graphWidget.setBackground('k')

          self.graphWidget.addLegend(offset=(-1, 1))

          self.measurement_scatter = self.graphWidget.plot(self.measurement_plot[:, 0], self.measurement_plot[:, 1], 
                                                           pen=None, symbol='o', name='Measurement')
          
          self.filtered_line = self.graphWidget.plot(self.filtered_plot[:, 0], self.filtered_plot[:, 1], 
                                                     pen=pg.mkPen('g', width=2), name='Filtered')
          
          self.actual_line =  self.graphWidget.plot(self.actual_plot[:, 0], self.actual_plot[:, 1], 
                                                    pen=pg.mkPen('r', width=2), name='Actual')

          self.timer = QtCore.QTimer()
          self.timer.setInterval(update_interval)
          self.timer.timeout.connect(self.update_plot_data)
          self.timer.start()

     def update_plot_data(self):
          try:
               self.actual = self.actual[1:, :]
               self.filtered = self.filtered[1:, :]
               self.measurements = self.measurements[1:, :]

               self.actual_plot = np.concatenate((self.actual_plot, [self.actual[0, :]]))
               self.filtered_plot = np.concatenate((self.filtered_plot, [self.filtered[0, :]]))
               self.measurement_plot = np.concatenate((self.measurement_plot, [self.measurements[0, :]]))
               
               self.actual_line.setData(self.actual_plot[:, 0], self.actual_plot[:, 1])
               self.filtered_line.setData(self.filtered_plot[:, 0], self.filtered_plot[:, 1])
               self.measurement_scatter.setData(self.measurement_plot[:, 0], self.measurement_plot[:, 1])
          except:
               pass



def live_plot(actual, measurements, filtered, x_range=[-20,120], y_range=[-1500,200], update_interval=10):
     app = QtWidgets.QApplication([])
     w = MainWindow(actual, measurements, filtered, x_range, y_range, update_interval)
     w.show()
     app.exec()
