import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
import random



class PlotKalman(QtWidgets.QMainWindow):

     def __init__(self, actual, measurements, filtered, x_range, y_range, update_interval, full_plot=False):
          super(PlotKalman, self).__init__()

          self.full_plot = full_plot
          self.count = 0

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
                                                           pen=None, symbol='x', name='Measurement')
          
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

               if self.count >= 50 and self.full_plot == False:
                    self.actual_plot = self.actual_plot[1:]
                    self.filtered_plot = self.filtered_plot[1:]
                    self.measurement_plot = self.measurement_plot[1:]
               
               self.count += 1

          except:
               pass



class PlotParticleFilter(QtWidgets.QMainWindow):

     def __init__(self, actual, measurements, pf, particles, x_range, y_range, update_interval, full_plot):
          super(PlotParticleFilter, self).__init__()

          self.full_plot = full_plot
          self.count = 0
          self.pf = pf

          self.graphWidget = pg.PlotWidget()
          self.graphWidget.setXRange(x_range[0], x_range[1])
          self.graphWidget.setYRange(y_range[0], y_range[1])
          self.setCentralWidget(self.graphWidget)

          self.actual = actual
          self.measurements = measurements
          self.particles = particles

          self.n_particles = particles.shape[0]

          self.actual_plot = np.array([[actual[0, 0], actual[0, 1]]])
          self.measurement_plot = np.array([[measurements[0, 0], measurements[0, 1]]])

          self.graphWidget.setBackground('k')
          self.graphWidget.addLegend(offset=(-1, 1))
          
          colour = lambda : random.randint(0, 255)

          self.particle_plots = []
          self.particle_lines = []
          particles = particles.tolist()
          for i in range(self.n_particles):
               self.particle_plots += [[particles[i][0]]]
               c = (colour(), colour(), colour())
               line = self.graphWidget.plot([self.particle_plots[i][0][0]], [self.particle_plots[i][0][1]],
                                             pen=pg.mkPen(c, width=0.2), symbol='o', symbolPen=c, symbolBrush=0.1, name=f'Particle {i+1}')
               self.particle_lines.append(line)

          self.measurement_scatter = self.graphWidget.plot(self.measurement_plot[:, 0], self.measurement_plot[:, 1], 
                                                           pen=None, symbol='x', symbolPen='w', symbolBrush=1, name='Measurement')

          self.actual_line =  self.graphWidget.plot(self.actual_plot[:, 0], self.actual_plot[:, 1], 
                                                    pen=pg.mkPen('w', width=5), name='Actual')
          

          self.timer = QtCore.QTimer()
          self.timer.setInterval(update_interval)
          self.timer.timeout.connect(self.update_plot_data)
          self.timer.start()

     def update_plot_data(self):
          try:
               self.actual = self.actual[1:, :]
               self.actual_plot = np.concatenate((self.actual_plot, [self.actual[0, :]]))
               self.actual_line.setData(self.actual_plot[:, 0], self.actual_plot[:, 1])

               self.measurements = self.measurements[1:, :]
               self.measurement_plot = np.concatenate((self.measurement_plot, [self.measurements[0, :]]))
               self.measurement_scatter.setData(self.measurement_plot[:, 0], self.measurement_plot[:, 1])
               
               self.particles = self.particles[:, 1:, :]
               for i in range(self.n_particles):
                    new_point = [self.particles[i, 0, :].tolist()]
                    # self.particle_plots[i] += new_point
                    self.particle_plots[i] = new_point
                    self.particle_lines[i].setData([point[0] for point in self.particle_plots[i]], [point[1] for point in self.particle_plots[i]])

               if self.count >= 25 and self.full_plot == False:
                    self.actual_plot = self.actual_plot[1:]
                    self.measurement_plot = self.measurement_plot[1:]
                    # for i in range(self.n_particles):
                    #      self.particle_plots[i] = self.particle_plots[i][1:]

               self.count += 1

               self.pf.reweight(self.measurements[-1, :])
               self.pf.resample()

          except:
               pass




def live_plot_kalman(actual, measurements, filtered, x_range=[-20,120], y_range=[-1500,200], update_interval=10, full_plot=False):
     app = QtWidgets.QApplication([])
     w = PlotKalman(actual, measurements, filtered, x_range, y_range, update_interval, full_plot)
     w.show()
     app.exec()


def live_plot_particle(actual, measurements, pf, particles, x_range=[-20,120], y_range=[-1500,200], update_interval=10, full_plot=False):
     app = QtWidgets.QApplication([])
     w = PlotParticleFilter(actual, measurements, pf, particles, x_range, y_range, update_interval, full_plot)
     w.show()
     app.exec()
