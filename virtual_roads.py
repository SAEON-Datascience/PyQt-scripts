import os
import sys
import time as time

import PyQt5
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import rasterio
import skimage
from skimage.graph import MCP_Geometric
from PyQt5 import QtCore, uic, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import gdal
import ogr
import osr

# PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
matplotlib.use("Qt5Agg")

UIClass, QtBaseClass = PyQt5.uic.loadUiType("mainwindow.ui")

roadlatitude = "global"
roadlongitude = "global"
roadvalues = "global"
roadvaluesX = "global"
roadvaluesY = "global"
road_coords = "global"

gridlatitude = "global"
gridlongitude = "global"
gridvalues = "global"
gridvaluesX = "global"
gridvaluesY = "global"
grid_coords = "global"

cost_surface = "global"
meters = "global"
road_surface = "global"
DEM = "global"
grid_surface = "global"
path_surface = "global"
start_time = "global"
start = "global"
fieldname = "global"
field_ids = "global"
output_dir = "global"
raster_dem = "global"
gdal_dir = "global"


class MainThread(QThread):
    valueChanged = PyQt5.QtCore.pyqtSignal(float)

    def __init__(self, uFilename, uFilename2, diagonal2, startpos2, endpos2):
        PyQt5.QtCore.QThread.__init__(self, parent=None)
        self.dempath = uFilename
        self.filename2 = uFilename2
        self.diagonal = diagonal2
        # self.filename3 = uFilename3
        self.exiting = True
        self.start_pos = startpos2
        self.end_pos = endpos2

        # self.na = nan

    def __del__(self):
        self.exiting = False
        self.wait()

    def run(self):
        self.sleep(1)  # Do "work"
        self.valueChanged.emit(-1)
        self.sleep(1)  # Do "work"
        # self.emit(QtCore.SIGNAL('__updateProgressBar(int)'), 0)  ## Reset progressbar value
        global gridlatitude
        global gridlongitude
        global gridvalues
        global gridvaluesX
        global gridvaluesY
        global grid_coords
        global roadlatitude
        global roadlongitude
        global roadvalues
        global roadvaluesX
        global roadvaluesY
        global meters
        global road_coords
        global cost_surface
        global path_surface
        global grid_surface
        global road_surface
        global gdal_dir

        def run2(k, dempath, outputpath, diagonal):
            global path_surface

            def pixelcoord(raster, xOffset, yOffset):

                geotransform = raster.GetGeoTransform()
                originX = geotransform[0]
                originY = geotransform[3]
                pixelWidth = geotransform[1]
                pixelHeight = geotransform[5]
                coordX = (originX + pixelWidth / 2) + pixelWidth * xOffset
                coordY = (originY + pixelHeight / 2) + pixelHeight * yOffset
                return coordX, coordY

            global gridlatitude
            global gridlongitude
            global gridvalues
            global gridvaluesX
            global gridvaluesY
            global grid_coords
            global roadlatitude
            global roadlongitude
            global roadvalues
            global roadvaluesX
            global roadvaluesY
            global meters
            global road_coords
            global cost_surface
            global path_surface
            global grid_surface
            global road_surface
            global raster_dem
            global gdal_dir

            gdal.UseExceptions()
            gdal.SetConfigOption("GDAL_DATA", gdal_dir)
            # start = time.time()
            # create temp cost surface
            path_surface2 = np.copy(cost_surface)
            path_surface2[path_surface2 > 0] = abs(
                path_surface2[gridvaluesY[k], gridvaluesX[k]] - path_surface2[path_surface2 > 0])
            path_surface2[path_surface2 > 0] = np.add(1, path_surface2[path_surface2 > 0])
            mcp = skimage.graph.MCP_Geometric(path_surface2, fully_connected=diagonal)
            cumulative_costs, traceback = mcp.find_costs([[gridvaluesY[k], gridvaluesX[k]]])  # start points
            cities = np.array(road_coords)  # end points
            ncities = cities.shape[0]
            paths = np.empty(path_surface2.shape)
            paths.fill(-1)
            costs2 = 0
            optimal_route = []
            i_val = 0
            x = sys.maxsize
            costsi = []
            costsa = []
            coordsa = []
            #
            raster = gdal.Open(dempath)
            try:
                for i in range(ncities):
                    try:

                        cost3 = cumulative_costs[cities[i][0], cities[i][1]]
                        if cost3 < x:
                            x = min(x, cost3)
                            i_val = i
                    except:
                        pass

                route = mcp.traceback([cities[i_val, :][0], cities[i_val, :][1]])
                optimal_route = route
                for j in range(len(route)):
                    costs2 += (meters + path_surface2[route[j]])
                    costsi.append(meters + path_surface2[route[j]])
                    ff2 = + (meters + path_surface2[route[j]])
                    costsa.append(ff2)
                    coordsa.append(pixelcoord(raster, route[j][1], route[j][0]))
                numList1 = [outputpath, "/", str(int(gridvalues[k])), ".txt"]
                numList2 = [outputpath, "/", str(int(gridvalues[k])), ".shp"]
                numList3 = [outputpath, "/", str(int(gridvalues[k])), ".prj"]
                filename1 = ''.join(numList1)
                filename2 = ''.join(numList2)
                filename3 = ''.join(numList3)
                file1 = open(filename1, "a")
                L = ["Lon", ",", "Lat", ",", "cost_dist(m)", ",", "accum_cost_dist(m)", "\n"]
                file1.writelines(''.join(L))
                for i in range(0, len(coordsa)):
                    L = [str(coordsa[i][0]), ", ", str(coordsa[i][1]), ", ", str(round(costsi[i], 3)), ", ",
                         str(round(costsa[i], 3)), "\n"]
                    file1.writelines(''.join(L))
                file1.close()
                driver = ogr.GetDriverByName('Esri Shapefile')
                ds = driver.CreateDataSource(filename2)
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                srs.MorphToESRI()
                file = open(filename3, 'w')
                file.write(srs.ExportToWkt())
                file.close()
                # ds.SetProjection(srs.ExportToWkt())
                layer = ds.CreateLayer('path', geom_type=ogr.wkbLineString)
                line = ogr.Geometry(ogr.wkbLineString)
                for i in range(0, len(coordsa)):
                    line.AddPoint(coordsa[i][0], coordsa[i][1])
                wkt = line.ExportToWkt()
                geom = ogr.CreateGeometryFromWkt(wkt)
                field_testfield = ogr.FieldDefn("dist_m", ogr.OFTReal)
                field_testfield.SetWidth(50)
                layer.CreateField(field_testfield)
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField("dist_m", costs2)
                feature.SetGeometry(geom)
                layer.CreateFeature(feature)
                feature = None
                ds = None
            except:
                pass
            raster = None

        # num_cores = multiprocessing.cpu_count()
        inputs = range(self.start_pos, self.end_pos)
        start2 = time.time()
        for k in inputs:
            run2(k, self.dempath, self.filename2, self.diagonal)
            self.sleep(0.01)  # Do "work"
            self.valueChanged.emit(round((k / len(inputs)) * 100, 2))

        self.valueChanged.emit(100)
        self.exiting = False
        # self.taskFinished.emit()


class process_grid_centroids(QThread):
    valueChanged = PyQt5.QtCore.pyqtSignal(float)

    def __init__(self, uFilename, uFilename2, uFilename3, nan):
        PyQt5.QtCore.QThread.__init__(self, parent=None)
        self.filename = uFilename
        self.filename2 = uFilename2
        self.filename3 = uFilename3
        self.exiting = True
        self.na = nan

    def __del__(self):
        self.exiting = False
        self.wait()

    def raster2array(self, rasterfn):
        raster = gdal.Open(rasterfn)
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        raster = None
        return array

    def pixelOffset2coord(self, raster, xOffset, yOffset):
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        coordX = (originX + pixelWidth / 2) + pixelWidth * xOffset
        coordY = (originY + pixelHeight / 2) + pixelHeight * yOffset
        return coordX, coordY

    def array2shp(self, array, rasterfn):

        gridlatitude = []
        gridlongitude = []
        gridvalues = []
        gridvaluesX = []
        gridvaluesY = []
        grid_coords = []

        # field_ids = []
        # max distance between points

        # source_ds = ogr.Open(self.filename2)
        # mb_l = source_ds.GetLayer()

        raster = gdal.Open(rasterfn)
        row_count = array.shape[0]
        for ridx, row in enumerate(array):
            if ridx % 100 == 0:
                # self.sleep(0.01)  # Do "work"
                self.valueChanged.emit(round((ridx / row_count) * 100, 2))  # Notify progress bar to update via signal
                # self.status.showMessage("Processing {} %".format(int((ridx/row_count)*100)))
                # self.progress.setValue(int((ridx/row_count)*100))
                # print("{0} of {1} rows processed".format(int((ridx/row_count)*100), (ridx/row_count)*100))
            for cidx, value in enumerate(row):
                if value >= 0:
                    Xcoord, Ycoord = self.pixelOffset2coord(raster, cidx, ridx)
                    gridlatitude.append(Ycoord)
                    gridlongitude.append(Xcoord)
                    gridvalues.append(int(value))
                    gridvaluesX.append(cidx)
                    gridvaluesY.append(ridx)
                    grid_coords.append([ridx, cidx])
                else:
                    pass
        raster = None
        return gridlatitude, gridlongitude, gridvalues, gridvaluesX, gridvaluesY, grid_coords

    def run(self):
        self.sleep(1)  # Do "work"
        self.valueChanged.emit(-1)
        self.sleep(1)  # Do "work"
        # self.emit(QtCore.SIGNAL('__updateProgressBar(int)'), 0)  ## Reset progressbar value
        global gridlatitude
        global gridlongitude
        global gridvalues
        global gridvaluesX
        global gridvaluesY
        global grid_coords
        global fieldname
        # process grid
        dempath = self.filename
        # print(dempath)
        ds = gdal.Open(dempath)
        # get extent
        GeoTransform = ds.GetGeoTransform()
        Projection = ds.GetProjection()
        x_min, xres, xskew, y_max, yskew, yres = GeoTransform
        x_max = x_min + (ds.RasterXSize * xres)
        y_min = y_max + (ds.RasterYSize * yres)
        x_res = ds.RasterXSize
        y_res = ds.RasterYSize
        pixel_width = xres
        ds = None
        # shpDriver = ogr.GetDriverByName("ESRI Shapefile")
        source_ds = ogr.Open(self.filename2)  # open the original shp
        mb_l = source_ds.GetLayer()
        # attrs = []
        # for i in range(mb_l.GetFeatureCount()):
        #     feature = mb_l.GetFeature(i)
        #     attrs.append(feature.GetField(fieldname))
        #     feature.Destroy()

        target_ds = gdal.GetDriverByName('GTiff').Create(self.filename3, x_res, y_res, 1, gdal.GDT_Float32)
        target_ds.SetGeoTransform(GeoTransform)
        target_ds.SetProjection(Projection)
        bandlist = target_ds.GetRasterBand(1)
        bandlist.SetNoDataValue(self.na)

        gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=%s" % fieldname])
        target_ds = None
        source_ds = None
        # get  empty array from road raster output
        array = self.raster2array(self.filename3)
        # self.initializing = False
        gridlatitude, gridlongitude, gridvalues, gridvaluesX, \
        gridvaluesY, grid_coords = self.array2shp(array, self.filename3)
        self.valueChanged.emit(100)
        self.exiting = False
        # self.taskFinished.emit()


class process_Obstacles(QThread):
    valueChanged = PyQt5.QtCore.pyqtSignal(float)

    def __init__(self, uFilename2, nan, x_res1, y_res1, GeoTransform1, Projection1):
        PyQt5.QtCore.QThread.__init__(self, parent=None)
        self.filename2 = uFilename2
        self.exiting = True
        self.na = nan
        self.x_res = x_res1
        self.y_res = y_res1
        self.GeoTransform = GeoTransform1
        self.Projection = Projection1

    def __del__(self):
        self.exiting = False
        self.wait()

    def run(self):
        self.sleep(1)  # Do "work"
        self.valueChanged.emit(-1)
        self.sleep(1)  # Do "work"
        # gdal.UseExceptions()
        # # path to gdal data directory
        # gdal.SetConfigOption("GDAL_DATA", "/Users/privateprivate/SAEON_data/DATA/gdal-data/")
        obstacles = []
        # output = "/Users/privateprivate/SAEON_data/DATA/obstacle_temp.tif"
        for root, dirs, files in os.walk(self.filename2):
            for file in files:
                if file.endswith(".shp"):
                    obstacles.append(os.path.join(root, file))

        row_count = len(obstacles)
        i = 0

        for file in obstacles:
            # self.sleep(1)
            self.valueChanged.emit(round((i / row_count) * 100, 2))
            # self.status.showMessage("Processing {} %".format(int((i / row_count) * 100)))
            # self.progress.setValue(int((i / row_count) * 100))
            shapefile1 = file
            source_ds = ogr.Open(shapefile1)
            mb_l = source_ds.GetLayer()

            mem_drv = gdal.GetDriverByName('MEM')
            target_ds = mem_drv.Create('', self.x_res, self.y_res, 1, gdal.GDT_Int16)
            # target_ds = gdal.GetDriverByName('GTiff').Create(self.filename3, self.x_res, self.y_res, 1, gdal.GDT_Int16)
            target_ds.SetGeoTransform(self.GeoTransform)
            target_ds.SetProjection(self.Projection)
            bandlist = target_ds.GetRasterBand(1)
            bandlist.SetNoDataValue(self.na)
            gdal.RasterizeLayer(target_ds, [1], mb_l, burn_values=[1])
            # target_ds = None
            # ds2 = gdal.Open(target_ds)
            obstacle_1 = np.array(target_ds.GetRasterBand(1).ReadAsArray())
            cost_surface[obstacle_1 == 1] = self.na
            target_ds = None
            # ds2 = None
            i += 1
        self.valueChanged.emit(100)
        self.exiting = False


class process_road_network(QThread):
    valueChanged = PyQt5.QtCore.pyqtSignal(float)

    def __init__(self, uFilename, uFilename2, uFilename3, nan):
        PyQt5.QtCore.QThread.__init__(self, parent=None)
        self.filename = uFilename
        self.filename2 = uFilename2
        self.filename3 = uFilename3
        self.exiting = True
        self.na = nan
        # self.now = datetime.datetime.now
        # self.target = self.now()

        # print(self.filename)

    def __del__(self):
        self.exiting = False
        self.wait()

    def raster2array(self, rasterfn):
        raster = gdal.Open(rasterfn)
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        raster = None
        return array

    def pixelOffset2coord(self, raster, xOffset, yOffset):
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        coordX = (originX + pixelWidth / 2) + pixelWidth * xOffset
        coordY = (originY + pixelHeight / 2) + pixelHeight * yOffset
        return coordX, coordY

    def array2shp(self, array, rasterfn):

        roadlatitude = []
        roadlongitude = []
        roadvalues = []
        roadvaluesX = []
        roadvaluesY = []
        road_coords = []
        # max distance between points
        raster = gdal.Open(rasterfn)
        row_count = array.shape[0]

        # one_second_later = datetime.timedelta(seconds=1)
        for ridx, row in enumerate(array):
            if ridx % 100 == 0:
                # self.sleep(0.01)  # Do "work"
                self.valueChanged.emit(round((ridx / row_count) * 100., 2))  # Notify progress bar to update via signal
                # end1 = time()
                # reverse = row_count-ridx
                # self.target += one_second_later
                # print(datetime.timedelta(seconds=reverse-1), 'remaining', end='\r')
                # self.sleep((self.target - self.now()).total_seconds())

            for cidx, value in enumerate(row):
                if value == 1:
                    Xcoord, Ycoord = self.pixelOffset2coord(raster, cidx, ridx)
                    roadlatitude.append(Ycoord)
                    roadlongitude.append(Xcoord)
                    roadvalues.append(value)
                    roadvaluesX.append(cidx)
                    roadvaluesY.append(ridx)
                    road_coords.append([ridx, cidx])
                else:
                    pass
        raster = None
        return roadlatitude, roadlongitude, roadvalues, roadvaluesX, roadvaluesY, road_coords

    def run(self):

        self.sleep(1)  # Do "work"
        self.valueChanged.emit(-1)
        self.sleep(1)  # Do "work"
        # self.emit(QtCore.SIGNAL('__updateProgressBar(int)'), 0)  ## Reset progressbar value
        global roadlatitude
        global roadlongitude
        global roadvalues
        global roadvaluesX
        global roadvaluesY
        global road_coords
        global start_time

        start_time = time.time()
        # process road network
        dempath = self.filename
        # print(dempath)
        ds = gdal.Open(dempath)
        # get extent
        GeoTransform = ds.GetGeoTransform()
        Projection = ds.GetProjection()
        x_min, xres, xskew, y_max, yskew, yres = GeoTransform
        x_max = x_min + (ds.RasterXSize * xres)
        y_min = y_max + (ds.RasterYSize * yres)
        x_res = ds.RasterXSize
        y_res = ds.RasterYSize
        pixel_width = xres
        ds = None
        source_ds = ogr.Open(self.filename2)  # open the original shp
        mb_l = source_ds.GetLayer()
        target_ds = gdal.GetDriverByName('GTiff').Create(self.filename3, x_res, y_res, 1, gdal.GDT_Float32)
        target_ds.SetGeoTransform(GeoTransform)
        target_ds.SetProjection(Projection)
        bandlist = target_ds.GetRasterBand(1)
        bandlist.SetNoDataValue(self.na)
        gdal.RasterizeLayer(target_ds, [1], mb_l, burn_values=[1])
        target_ds = None
        # get  empty array from road raster output
        array = self.raster2array(self.filename3)
        # self.initializing = False
        roadlatitude, roadlongitude, roadvalues, roadvaluesX, \
        roadvaluesY, road_coords = self.array2shp(array, self.filename3)
        self.valueChanged.emit(100)
        self.exiting = False
        # self.taskFinished.emit()


class MyApp(UIClass, QtBaseClass):

    def getItem(self, items, message):
        item, ok = QInputDialog.getItem(self, "select input dialog",
                                        message, items, 0, False)
        if ok and item:
            return item

    def degrees_to_meters(self, lat):
        import math
        radius = 6371.0088
        lat1 = 0. * math.pi / 180.
        lat2 = 0. * math.pi / 180.
        lon1 = lat * math.pi / 180.
        lon2 = 0. * math.pi / 180.
        # dlat = lat2 - lat1
        dlon = lon2 - lon1
        rad = math.acos(
            (math.sin(lat1) * math.sin(lat2))
            +
            (math.cos(lat2) * math.cos(dlon))
        )
        answerlon = (radius * 1000) * rad;
        if lat < 0:
            answerlon = answerlon - answerlon - answerlon
        return (answerlon)

    def calculate_slope(self, DEM):
        gdal.DEMProcessing('slope.tif', DEM, 'slope', scale=111120)
        ds = gdal.Open('slope.tif')
        slope = np.array(ds.GetRasterBand(1).ReadAsArray())
        ds = None
        return slope


        # with rasterio.open('slope.tif') as dataset:
        #     slope = dataset.read(1)
        # return slope

    def update_quickview(self, image, image2, image3, image4):
        # PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        scene = PyQt5.QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(scene)
        figure = Figure()
        figure.clear()
        # axes = figure.gca()
        axes = figure.add_subplot(111)
        axes.axis('off')

        im1 = axes.imshow(image, interpolation='none', aspect='auto')
        if image2:
            im1 = axes.imshow(image, interpolation='none', aspect='auto', alpha=0.75)
            im2 = axes.imshow(grid_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('Reds'), alpha=1)
        if image3:
            im1 = axes.imshow(image, interpolation='none', aspect='auto', alpha=0.5)
            im2 = axes.imshow(grid_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('Reds'), alpha=0.75)
            im3 = axes.imshow(road_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('gray'), alpha=1)
        # if image4:
        #     im1 = axes.imshow(image, interpolation='none', aspect='auto', alpha=0.5)
        #     im2 = axes.imshow(grid_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('Reds'), alpha=0.75)
        #     im3 = axes.imshow(road_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('gray'), alpha=1)
        #     im4 = axes.imshow(path_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('spring'), alpha=1)

        figure.colorbar(im1)
        # figure.colorbar(im)
        canvas = FigureCanvas(figure)
        width = self.graphicsView.frameRect().width()
        height = self.graphicsView.frameRect().height()
        # print(width)
        canvas.setGeometry(0, 0, width - 10, height - 10)
        canvas.draw()
        scene.addWidget(canvas)

    def updateProgressBar(self, maxVal):
        if maxVal == -1:
            self.progressBar.setRange(0, 0)
        elif maxVal == 100:
            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(0)
        else:
            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(maxVal)
        self.repaint()

    def is_whole(self, n):
        return n % 1 == 0

    def updateStatusBar(self, maxVal):
        global start
        fmt = "  Progress: {:>3}%. Time: {:>3}s"
        fmt2 = "  Progress: {:>3}%. Time: {:>3}m"
        fmt3 = "  Progress: {:>3}%. Time: {:>3}m"
        fmt4 = "  Progress: {:>3}%. Time: {:>3}m"
        num = 100

        if maxVal == -1:
            start = time.time()
            # start = start.astype(float)
            self.statusBar().showMessage("Initializing...")
        elif maxVal == 100:
            stop = time.time()
            remaining = round((stop - start), 2)
            if remaining < 60:
                self.statusBar().showMessage("Ready - last action took {:>3}s".format(remaining))
                # self.statusBar().showMessage(fmt.format(maxVal, remaining))
            elif 60 < remaining < 3600:
                self.statusBar().showMessage("Ready - last action took {:>3}m".format(round(remaining / 60, 2)))
                # self.statusBar().showMessage(fmt2.format(maxVal, round(remaining, 2)))
            elif 3600 < remaining < 86400:
                self.statusBar().showMessage("Ready - last action took {:>3}h".format(round(remaining / 60 / 60, 2)))
                # self.statusBar().showMessage(fmt3.format(maxVal, round(remaining, 2)))
            else:
                self.statusBar().showMessage(
                    "Ready - last action took {:>3}d".format(round(remaining / 60 / 60 / 24, 2)))
                # self.statusBar().showMessage(fmt4.format(maxVal, round(remaining, 2)))
            # self.statusBar().showMessage(fmt.format(maxVal, remaining))


        else:
            i = maxVal
            stop = time.time()
            # stop = stop.astype(float)
            # diff=stop-start

            remaining = round((stop - start), 2)
            if remaining < 60:
                self.statusBar().showMessage(fmt.format(maxVal, remaining))
            elif remaining > 60 and remaining < 3600:
                self.statusBar().showMessage(fmt2.format(maxVal, round(remaining / 60, 2)))
            elif remaining > 3600 and remaining < 86400:
                self.statusBar().showMessage(fmt3.format(maxVal, round(remaining / 60 / 60, 2)))
            else:
                self.statusBar().showMessage(fmt4.format(maxVal, round(remaining / 60 / 60 / 24, 2)))
            # else:
            #     self.statusBar().showMessage("Progress: {:>3}%".format(maxVal))
            #     #print(fmt.format(100 * i // num, remaining), end='\r')

        self.repaint()

    def update_pathfinding_quickview(self, maxVal):
        if maxVal == -1:
            pass
        elif maxVal == 100:
            self.update_quickview(cost_surface, True, True, True)
        else:
            self.update_quickview(cost_surface, True, True, True)
        self.repaint()

    def select_gdal_data_dir(self):
        global output_dir
        global gdal_dir
        # gdal.SetConfigOption("GDAL_DATA", "/Users/privateprivate/SAEON_data/DATA/gdal-data/")
        filename = QFileDialog.getExistingDirectory(None, 'Select directory', os.sep.join(
            (os.path.expanduser('~'))), QFileDialog.ShowDirsOnly)
        if filename:
            self.lineEdit_9.setText(str(filename))
            gdal_dir = str(filename)
            gdal.UseExceptions()
            # # path to gdal data directory
            gdal.SetConfigOption("GDAL_DATA", str(filename))

            self.lineEdit_9.setCursorPosition(0)
            self.lineEdit_8.setEnabled(True)
            self.pushButton_7.setEnabled(True)
            time.sleep(1)

    def select_output_dir(self):
        global output_dir
        # gdal.SetConfigOption("GDAL_DATA", "/Users/privateprivate/SAEON_data/DATA/gdal-data/")
        filename = QFileDialog.getExistingDirectory(None, 'Select directory', os.sep.join(
            (os.path.expanduser('~'))), QFileDialog.ShowDirsOnly)
        if filename:
            self.lineEdit_8.setText(str(filename))
            self.lineEdit_8.setCursorPosition(0)
            self.lineEdit_8.setEnabled(True)
            self.pushButton.setEnabled(True)
            time.sleep(1)

    def select_DEM_file(self):
        global cost_surface
        global meters
        global DEM
        # PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

        filename, _filter = QFileDialog.getOpenFileName(None, 'Choose tif', os.sep.join(
            (os.path.expanduser('~'))),
                                                        'Geotiff files (*.tif)')
        if filename:
            self.lineEdit.setText(str(filename))
            self.lineEdit.setCursorPosition(0)
            ds =gdal.Open(str(filename))
            cost_surface = np.array(ds.GetRasterBand(1).ReadAsArray())

            self.GeoTransform = ds.GetGeoTransform()
            self.Projection = ds.GetProjection()
            self.x_min, self.xres, self.xskew, self.y_max, self.yskew, self.yres = self.GeoTransform
            self.x_max = self.x_min + (ds.RasterXSize * self.xres)
            self.y_min = self.y_max + (ds.RasterYSize * self.yres)
            self.x_res = ds.RasterXSize
            self.y_res = ds.RasterYSize
            self.pixel_width = self.xres
            meters = self.degrees_to_meters(self.pixel_width)
            ds = None
            slope = self.calculate_slope(str(filename))
            slope[slope < 0] = 0
            slope[slope >= self.doubleSpinBox.value()] = self.na  # na values
            cost_surface[slope == self.na] = self.na

            self.pushButton_3.setEnabled(True)
            self.pushButton_6.setEnabled(True)
            self.pushButton_4.setEnabled(True)
            self.checkBox.setEnabled(True)
            self.checkBox.setChecked(True)
            self.checkBox_3.setChecked(False)
            self.checkBox_4.setChecked(False)
            # self.checkBox_5.setChecked(False)
            dempath = self.lineEdit.text()
            ds = gdal.Open(dempath)
            DEM = np.array(ds.GetRasterBand(1).ReadAsArray())
            ds = None
            self.update_quickview(cost_surface, None, None, None)

    def load_obstacles(self):
        global cost_surface
        filename = QFileDialog.getExistingDirectory(None, 'Select directory', os.sep.join(
            (os.path.expanduser('~'))), QFileDialog.ShowDirsOnly)
        if filename:
            self.lineEdit_4.setText(str(filename))
            self.lineEdit_4.setCursorPosition(0)

            self.progressBar.setValue(0)
            self.statusbar.showMessage("Initializing...")
            # start = perf_counter()
            # self.progressBar.setRange(0,100)
            # output = "/Users/privateprivate/SAEON_data/DATA/obstacle_temp.tif"
            self.mm = process_Obstacles(
                str(filename), self.na, self.x_res, self.y_res,
                self.GeoTransform, self.Projection
            )
            self.mm.start()
            self.mm.valueChanged.connect(self.updateProgressBar)
            self.mm.valueChanged.connect(self.updateStatusBar)
            while self.mm.exiting:
                QCoreApplication.processEvents()

            self.progressBar.setValue(0)
            # self.pushButton_5.setEnabled(True)
            # self.checkBox_2.setChecked(False)
            # self.checkBox.setEnabled(True)
            # self.checkBox.setChecked(True)
            self.update_quickview(cost_surface, None, None, None)
            time.sleep(1)

    def load_centroids(self):
        global gridlatitude
        global gridlongitude
        global gridvalues
        global gridvaluesX
        global gridvaluesY
        global grid_coords
        global cost_surface
        global grid_surface
        global fieldname
        global field_ids
        filename, _filter = QFileDialog.getOpenFileName(None, 'Choose shapefile', os.sep.join(
            (os.path.expanduser('~'))),
                                                        'Shapefile (*.shp)')
        if filename:
            self.lineEdit_5.setText(str(filename))
            self.lineEdit_5.setCursorPosition(0)

            source_ds = ogr.Open(str(filename))
            mb_l = source_ds.GetLayer()
            schema = []
            ldefn = mb_l.GetLayerDefn()
            for n in range(ldefn.GetFieldCount()):
                fdefn = ldefn.GetFieldDefn(n)
                schema.append(fdefn.name)
            # print(schema)
            source_ds = None

            items = schema  # ("C", "C++", "Java", "Python", "marc")
            fieldname = self.getItem(items, "select unique field")
            self.lineEdit_7.setText(fieldname)
            grid_centroid_out = str(filename).replace(str(filename)[str(filename).rfind('.'):len(str(filename))],
                                                      "_processed.tif")

            self.progressBar.setRange(0, 100)
            dempath = self.lineEdit.text()
            self.mm = process_grid_centroids(
                dempath, str(filename), grid_centroid_out, self.na
            )
            self.mm.start()
            self.mm.valueChanged.connect(self.updateProgressBar)
            self.mm.valueChanged.connect(self.updateStatusBar)
            while self.mm.exiting:
                QCoreApplication.processEvents()

            self.progressBar.setValue(0)
            # self.statusbar.showMessage("Ready")
            self.lineEdit_6.setText(grid_centroid_out)
            self.lineEdit_6.setCursorPosition(0)

            # temp=cost_surface
            ds = gdal.Open(grid_centroid_out)
            grid_surface = np.array(ds.GetRasterBand(1).ReadAsArray())
            ds = None
            grid_surface[grid_surface < 0.] = float('nan')
            # temp[grid_surface > 0] = 0

            self.checkBox.setChecked(False)
            self.checkBox_3.setEnabled(True)
            self.checkBox_3.setChecked(True)
            self.checkBox_4.setChecked(False)
            # self.checkBox_5.setChecked(False)

            self.update_quickview(cost_surface, True, False, False)
            self.pushButton_5.setEnabled(True)

            self.comboBox.clear()
            for i in range(0, len(gridvalues)):
                self.comboBox.addItem("{} ({})".format(i, str(gridvalues[i])))
            self.comboBox.setEnabled(True)
            self.label_11.setText("{} {}".format(len(gridvalues), "positions"))
            self.lineEdit_10.setText("0")
            self.lineEdit_10.setCursorPosition(0)
            self.lineEdit_11.setText(str(len(gridvalues)))
            self.lineEdit_11.setCursorPosition(0)
            time.sleep(1)

    def select_ROAD_file(self):
        global roadlatitude
        global roadlongitude
        global roadvalues
        global roadvaluesX
        global roadvaluesY
        global road_coords
        global cost_surface
        global road_surface
        global start_time
        global start

        # gdal.UseExceptions()
        # # path to gdal data directory
        # gdal.SetConfigOption("GDAL_DATA", "/Users/privateprivate/SAEON_data/DATA/gdal-data/")
        filename, _filter = QFileDialog.getOpenFileName(None, 'Choose shapefile', os.sep.join(
            (os.path.expanduser('~'))),
                                                        'Shapefile (*.shp)')
        if filename:
            # road_network_shp = str(filename)
            self.lineEdit_2.setText(str(filename))
            self.lineEdit_2.setCursorPosition(0)
            road_network_out = str(filename).replace(str(filename)[str(filename).rfind('.'):len(str(filename))],
                                                     "_processed.tif")

            self.progressBar.setRange(0, 100)
            dempath = self.lineEdit.text()
            self.mm = process_road_network(
                dempath, str(filename), road_network_out, self.na
            )
            self.mm.start()
            self.mm.valueChanged.connect(self.updateProgressBar)
            self.mm.valueChanged.connect(self.updateStatusBar)
            while self.mm.exiting:
                QCoreApplication.processEvents()
            self.progressBar.setValue(0)
            self.statusbar.showMessage("Ready")
            self.lineEdit_3.setText(road_network_out)
            self.lineEdit_3.setCursorPosition(0)
            # temp=cost_surface
            ds = gdal.Open(road_network_out)
            road_surface = np.array(ds.GetRasterBand(1).ReadAsArray())
            ds = None
            road_surface[road_surface < 0.] = float('nan')
            self.checkBox.setChecked(False)
            self.checkBox_4.setEnabled(True)
            self.checkBox_4.setChecked(True)
            self.checkBox_3.setChecked(False)
            # self.checkBox_5.setChecked(False)

            self.update_quickview(cost_surface, True, True, False)
            self.pushButton_2.setEnabled(True)
            self.pushButton_8.setEnabled(True)
            time.sleep(1)

    def run2(self):
        from joblib import Parallel, delayed
        import multiprocessing

        def run(k, dempath, outputpath, diagonal):
            def pixelcoord(raster, xOffset, yOffset):

                geotransform = raster.GetGeoTransform()
                originX = geotransform[0]
                originY = geotransform[3]
                pixelWidth = geotransform[1]
                pixelHeight = geotransform[5]
                coordX = (originX + pixelWidth / 2) + pixelWidth * xOffset
                coordY = (originY + pixelHeight / 2) + pixelHeight * yOffset
                return coordX, coordY

            global gridlatitude
            global gridlongitude
            global gridvalues
            global gridvaluesX
            global gridvaluesY
            global grid_coords
            global roadlatitude
            global roadlongitude
            global roadvalues
            global roadvaluesX
            global roadvaluesY
            global meters
            global road_coords
            global cost_surface
            global path_surface
            global grid_surface
            global road_surface
            global raster_dem
            global gdal_dir

            gdal.UseExceptions()
            gdal.SetConfigOption("GDAL_DATA", gdal_dir)
            # start = time.time()
            # create temp cost surface
            path_surface2 = np.copy(cost_surface)
            path_surface2[path_surface2 > 0] = abs(
                path_surface2[gridvaluesY[k], gridvaluesX[k]] - path_surface2[path_surface2 > 0])
            path_surface2[path_surface2 > 0] = np.add(1, path_surface2[path_surface2 > 0])
            mcp = skimage.graph.MCP_Geometric(path_surface2, fully_connected=diagonal)
            cumulative_costs, traceback = mcp.find_costs([[gridvaluesY[k], gridvaluesX[k]]])  # start points
            cities = np.array(road_coords)  # end points
            ncities = cities.shape[0]
            paths = np.empty(path_surface2.shape)
            paths.fill(-1)
            costs2 = 0
            optimal_route = []
            i_val = 0
            x = sys.maxsize
            costsi = []
            costsa = []
            coordsa = []
            #
            raster = gdal.Open(dempath)
            try:
                for i in range(ncities):
                    try:

                        cost3 = cumulative_costs[cities[i][0], cities[i][1]]
                        if cost3 < x:
                            x = min(x, cost3)
                            i_val = i
                    except:
                        pass

                route = mcp.traceback([cities[i_val, :][0], cities[i_val, :][1]])
                optimal_route = route
                # path_surface2[path_surface2 > 0] = abs(
                #     path_surface2[gridvaluesY[k], gridvaluesX[k]] - path_surface2[path_surface2 > 0])
                for j in range(len(route)):
                    costs2 += (meters + path_surface2[route[j]])
                    costsi.append(meters + path_surface2[route[j]])
                    ff2 = + (meters + path_surface2[route[j]])
                    costsa.append(ff2)
                    coordsa.append(pixelcoord(raster, route[j][1], route[j][0]))

                numList1 = [outputpath, "/", str(int(gridvalues[k])), ".txt"]
                numList2 = [outputpath, "/", str(int(gridvalues[k])), ".shp"]
                numList3 = [outputpath, "/", str(int(gridvalues[k])), ".prj"]
                filename1 = ''.join(numList1)
                filename2 = ''.join(numList2)
                filename3 = ''.join(numList3)
                file1 = open(filename1, "a")
                L = ["Lon", ",", "Lat", ",", "cost_dist(m)", ",", "accum_cost_dist(m)", "\n"]
                file1.writelines(''.join(L))
                for i in range(0, len(coordsa)):
                    L = [str(coordsa[i][0]), ", ", str(coordsa[i][1]), ", ", str(round(costsi[i], 3)), ", ",
                         str(round(costsa[i], 3)), "\n"]
                    file1.writelines(''.join(L))
                file1.close()
                driver = ogr.GetDriverByName('Esri Shapefile')
                ds = driver.CreateDataSource(filename2)
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                srs.MorphToESRI()
                file = open(filename3, 'w')
                file.write(srs.ExportToWkt())
                file.close()
                # ds.SetProjection(srs.ExportToWkt())
                layer = ds.CreateLayer('path', geom_type=ogr.wkbLineString)
                line = ogr.Geometry(ogr.wkbLineString)
                for i in range(0, len(coordsa)):
                    line.AddPoint(coordsa[i][0], coordsa[i][1])
                wkt = line.ExportToWkt()
                geom = ogr.CreateGeometryFromWkt(wkt)
                field_testfield = ogr.FieldDefn("dist_m", ogr.OFTReal)
                field_testfield.SetWidth(50)
                layer.CreateField(field_testfield)
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField("dist_m", costs2)
                feature.SetGeometry(geom)
                layer.CreateFeature(feature)
                feature = None
                ds = None
            except:
                pass
            raster = None
            # stop = time.time()
            # remaining = round((stop - start), 2)
            # print(remaining)
            # return(optimal_route)

        if self.checkBox_7.isChecked():
            self.progressBar.setRange(0, 0)
            num_cores = multiprocessing.cpu_count()

            # for i in inputs:
            #     self.run(i)
            dempath = self.lineEdit.text()
            # raster_dem = gdal.Open(dempath)
            start2 = time.time()
            selectedLayerIndexs = int(str(self.lineEdit_10.text()))
            selectedLayerIndexe = int(str(self.lineEdit_11.text()))
            inputs = range(selectedLayerIndexs, selectedLayerIndexe)
            time.sleep(1)
            self.progressBar.setRange(0, 100)
            self.progressBar.setValue(25)
            self.updateStatusBar(-1)
            self.updateProgressBar(-1)
            time.sleep(1)
            if (selectedLayerIndexe - selectedLayerIndexs) + 1 < num_cores:
                num_cores = (selectedLayerIndexe - selectedLayerIndexs) + 1
            for i in range(selectedLayerIndexs - 1, selectedLayerIndexe, num_cores):
                inputs = range(i + 1, i + num_cores)
                Parallel(n_jobs=num_cores)(
                    delayed(run)(k, dempath, self.lineEdit_8.text(), self.checkBox_7.isChecked()) for k in inputs)
                time.sleep(1)
                self.updateStatusBar(int(i + num_cores) / int(selectedLayerIndexe))
                self.updateProgressBar(int(i + num_cores) / int(selectedLayerIndexe))
            time.sleep(1)
            self.updateStatusBar(100)
            self.updateProgressBar(100)

            # for k in inputs:
            #     run(k,dempath,self.lineEdit_8.text())

            stop2 = time.time()
            remaining2 = round((stop2 - start2), 2)
            # print("")
            # print(remaining2)
        else:
            self.progressBar.setRange(0, 100)
            dempath = self.lineEdit.text()
            selectedLayerIndexs = int(str(self.lineEdit_10.text()))
            selectedLayerIndexe = int(str(self.lineEdit_11.text()))
            self.mm = MainThread(
                dempath, self.lineEdit_8.text(), self.checkBox_7.isChecked, selectedLayerIndexs, selectedLayerIndexe
            )
            self.mm.start()
            self.mm.valueChanged.connect(self.updateProgressBar)
            self.mm.valueChanged.connect(self.updateStatusBar)
            while self.mm.exiting:
                QCoreApplication.processEvents()

            self.progressBar.setValue(0)

    def check_path(self):
        def runn(k, dempath, outputpath, diagonal):
            print(diagonal)
            def pixelcoord(raster, xOffset, yOffset):

                geotransform = raster.GetGeoTransform()
                originX = geotransform[0]
                originY = geotransform[3]
                pixelWidth = geotransform[1]
                pixelHeight = geotransform[5]
                coordX = (originX + pixelWidth / 2) + pixelWidth * xOffset
                coordY = (originY + pixelHeight / 2) + pixelHeight * yOffset
                return coordX, coordY

            global gridlatitude
            global gridlongitude
            global gridvalues
            global gridvaluesX
            global gridvaluesY
            global grid_coords
            global roadlatitude
            global roadlongitude
            global roadvalues
            global roadvaluesX
            global roadvaluesY
            global meters
            global road_coords
            global cost_surface
            global path_surface
            global grid_surface
            global road_surface
            global raster_dem
            global gdal_dir

            gdal.UseExceptions()
            gdal.SetConfigOption("GDAL_DATA", gdal_dir)
            start = time.time()
            # create temp cost surface
            path_surface2 = np.copy(cost_surface)
            path_surface2[path_surface2 > 0] = abs(
                path_surface2[gridvaluesY[k], gridvaluesX[k]] - path_surface2[path_surface2 > 0])
            path_surface2[path_surface2 > 0] = np.add(1, path_surface2[path_surface2 > 0])
            mcp = skimage.graph.MCP_Geometric(path_surface2, fully_connected=diagonal)
            cumulative_costs, traceback = mcp.find_costs([[gridvaluesY[k], gridvaluesX[k]]])  # start points
            cities = np.array(road_coords)  # end points
            ncities = cities.shape[0]
            paths = np.empty(path_surface2.shape)
            paths.fill(-1)
            costs2 = 0
            optimal_route = []
            i_val = 0
            x = sys.maxsize
            costsi = []
            costsa = []
            coordsa = []
            #
            raster = gdal.Open(dempath)
            try:
                for i in range(ncities):
                    try:

                        cost3 = cumulative_costs[cities[i][0], cities[i][1]]
                        if cost3 < x:
                            x = min(x, cost3)
                            i_val = i
                    except:
                        pass

                route = mcp.traceback([cities[i_val, :][0], cities[i_val, :][1]])
                optimal_route = route
                # path_surface2[path_surface2 > 0] = abs(
                #     path_surface2[gridvaluesY[k], gridvaluesX[k]] - path_surface2[path_surface2 > 0])
                for j in range(len(route)):
                    costs2 += (meters + path_surface2[route[j]])
                    costsi.append(meters + path_surface2[route[j]])
                    ff2 = + (meters + path_surface2[route[j]])
                    costsa.append(ff2)
                    coordsa.append(pixelcoord(raster, route[j][1], route[j][0]))
                # path_surface = np.copy(cost_surface)
                # path_surface[optimal_route]=
                # path_surface = np.copy(DEM)
                # path_surface=path_surface.astype(float)
                # for i in optimal_route:
                #     path_surface[i]=-1000.
                # path_surface[path_surface> -1000.]=float('nan')
                # path_surface[path_surface == -1000.] = 2000

                numList1 = [outputpath, "/", str(int(gridvalues[k])), ".txt"]
                numList2 = [outputpath, "/", str(int(gridvalues[k])), ".shp"]
                numList3 = [outputpath, "/", str(int(gridvalues[k])), ".prj"]
                filename1 = ''.join(numList1)
                filename2 = ''.join(numList2)
                filename3 = ''.join(numList3)
                file1 = open(filename1, "a")
                L = ["Lon", ",", "Lat", ",", "cost_dist(m)", ",", "accum_cost_dist(m)", "\n"]
                file1.writelines(''.join(L))
                for i in range(0, len(coordsa)):
                    L = [str(coordsa[i][0]), ", ", str(coordsa[i][1]), ", ", str(round(costsi[i], 3)), ", ",
                         str(round(costsa[i], 3)), "\n"]
                    file1.writelines(''.join(L))
                file1.close()
                driver = ogr.GetDriverByName('Esri Shapefile')
                ds = driver.CreateDataSource(filename2)
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                srs.MorphToESRI()
                file = open(filename3, 'w')
                file.write(srs.ExportToWkt())
                file.close()
                # ds.SetProjection(srs.ExportToWkt())
                layer = ds.CreateLayer('path', geom_type=ogr.wkbLineString)
                line = ogr.Geometry(ogr.wkbLineString)
                for i in range(0, len(coordsa)):
                    line.AddPoint(coordsa[i][0], coordsa[i][1])
                wkt = line.ExportToWkt()
                geom = ogr.CreateGeometryFromWkt(wkt)
                field_testfield = ogr.FieldDefn("dist_m", ogr.OFTReal)
                field_testfield.SetWidth(50)
                layer.CreateField(field_testfield)
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField("dist_m", costs2)
                feature.SetGeometry(geom)
                layer.CreateFeature(feature)
                feature = None
                ds = None
            except:
                pass
            raster = None
            stop = time.time()
            remaining = round((stop - start), 2)
            print(remaining)
            # return(optimal_route)

        fmt = "  Progress: {:>3}%. Time: {:>3}s"
        fmt2 = "  Progress: {:>3}%. Time: {:>3}m"
        fmt3 = "  Progress: {:>3}%. Time: {:>3}m"
        fmt4 = "  Progress: {:>3}%. Time: {:>3}m"
        num = 100

        dempath = self.lineEdit.text()
        # raster_dem = gdal.Open(dempath)
        start2 = time.time()



        selectedLayerIndexs = int(str(self.lineEdit_10.text()))
        selectedLayerIndexe = int(str(self.lineEdit_11.text()))
        inputs = range(selectedLayerIndexs, selectedLayerIndexe)
        selectedLayerIndex = self.comboBox.currentIndex()
        time.sleep(1)
        self.progressBar.setRange(0, 100)
        inputs = range(selectedLayerIndexs, selectedLayerIndexe)

        runn(selectedLayerIndex, dempath, self.lineEdit_8.text(), self.checkBox_8.isChecked())
        stop = time.time()
        remaining = round((stop - start), 2)
        if remaining < 60:
            self.statusBar().showMessage("Ready - last action took {:>3}s".format(remaining))
            # self.statusBar().showMessage(fmt.format(maxVal, remaining))
        elif 60 < remaining < 3600:
            self.statusBar().showMessage("Ready - last action took {:>3}m".format(round(remaining / 60, 2)))
            # self.statusBar().showMessage(fmt2.format(maxVal, round(remaining, 2)))
        elif 3600 < remaining < 86400:
            self.statusBar().showMessage("Ready - last action took {:>3}h".format(round(remaining / 60 / 60, 2)))
            # self.statusBar().showMessage(fmt3.format(maxVal, round(remaining, 2)))
        else:
            self.statusBar().showMessage(
                "Ready - last action took {:>3}d".format(round(remaining / 60 / 60 / 24, 2)))

    def plot(self):
        global DEM
        global cost_surface
        global road_surface
        global grid_surface
        global path_surface
        # PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        try:
            if self.checkBox.isChecked():
                matplotlib.rcParams['figure.figsize'] = (8, 5.5)
                # plt.imshow(DEM, interpolation='none', aspect='auto')
                plt.imshow(cost_surface, interpolation='none', aspect='auto')
                plt.colorbar()
                plt.show()
            elif self.checkBox_3.isChecked():
                matplotlib.rcParams['figure.figsize'] = (8, 5.5)
                fig = plt.figure()
                axes = fig.add_subplot(111)
                im1 = axes.imshow(cost_surface, interpolation='none', aspect='auto')
                im2 = axes.imshow(grid_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('Reds'))

                fig.colorbar(im1)
                plt.show()
            elif self.checkBox_4.isChecked():
                matplotlib.rcParams['figure.figsize'] = (8, 5.5)
                fig = plt.figure()
                axes = fig.add_subplot(111)
                im1 = axes.imshow(cost_surface, interpolation='none', aspect='auto')
                im2 = axes.imshow(road_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('Reds'))
                im3 = axes.imshow(grid_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('gray'))

                fig.colorbar(im1)
                plt.show()
            # elif self.checkBox_5.isChecked():
            #     matplotlib.rcParams['figure.figsize'] = (8, 5.5)
            #     fig = plt.figure()
            #     axes = fig.add_subplot(111)
            #     im1 = axes.imshow(cost_surface, interpolation='none', aspect='auto')
            #     im2 = axes.imshow(road_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('Reds'))
            #     im3 = axes.imshow(grid_surface, interpolation='none', aspect='auto', cmap=plt.get_cmap('gray'))
            #     im4 = axes.imshow(path_surface, interpolation='none', aspect='auto',cmap=plt.get_cmap('spring'))
            #     fig.colorbar(im1)
            #     plt.show()
        except:
            pass

    def checkBox_cost_surface(self):
        global cost_surface
        self.checkBox.setChecked(True)
        if self.checkBox.isChecked:
            # self.checkBox_2.setChecked(False)
            self.checkBox_3.setChecked(False)
            self.checkBox_4.setChecked(False)
            # self.checkBox_5.setChecked(False)
            self.update_quickview(cost_surface, None, None, None)

    def checkBox_start_surface(self):
        global cost_surface
        global road_surface
        global grid_surface
        self.checkBox_3.setChecked(True)
        if self.checkBox_3.isChecked:
            # self.checkBox_2.setChecked(False)
            self.checkBox.setChecked(False)
            self.checkBox_4.setChecked(False)
            # self.checkBox_5.setChecked(False)
            self.update_quickview(cost_surface, True, None, None)

    def checkBox_start_and_end_surface(self):
        global cost_surface
        global road_surface
        global grid_surface
        self.checkBox_4.setChecked(True)
        if self.checkBox_4.isChecked:
            # self.checkBox_2.setChecked(False)
            self.checkBox.setChecked(False)
            self.checkBox_3.setChecked(False)
            # self.checkBox_5.setChecked(False)
            self.update_quickview(cost_surface, True, True, None)

    def checkBox_save_as_txt(self):
        self.checkBox_6.setChecked(True)
        if self.checkBox_6.isChecked:
            self.checkBox_7.setChecked(False)
            # self.checkBox_8.setChecked(False)

    def checkBox_save_as_shp(self):
        self.checkBox_7.setChecked(True)
        if self.checkBox_7.isChecked:
            self.checkBox_6.setChecked(False)
            # self.checkBox_8.setChecked(False)

    def __init__(self):
        # PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        global start
        UIClass.__init__(self)
        QtBaseClass.__init__(self)
        self.setupUi(self)

        self.pushButton_9.clicked.connect(self.select_gdal_data_dir)
        self.pushButton_7.clicked.connect(self.select_output_dir)
        self.pushButton.clicked.connect(self.select_DEM_file)
        self.pushButton_3.clicked.connect(self.plot)
        #
        self.pushButton_4.clicked.connect(self.load_obstacles)

        # self.checkBox_2.clicked.connect(self.checkBox_DEMonly)
        self.checkBox.clicked.connect(self.checkBox_cost_surface)
        self.checkBox_3.clicked.connect(self.checkBox_start_surface)
        self.checkBox_4.clicked.connect(self.checkBox_start_and_end_surface)
        # self.checkBox_5.clicked.connect(self.checkBox_path_surface)

        self.checkBox_6.clicked.connect(self.checkBox_save_as_txt)
        self.checkBox_7.clicked.connect(self.checkBox_save_as_shp)

        self.progressBar = QProgressBar()
        self.statusBar().addPermanentWidget(self.progressBar)
        self.progressBar.setGeometry(30, 40, 200, 25)
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        # s#elf.StatusBar() = self.statusbar
        self.statusbar.showMessage("Ready")
        # self.cost_surface=None
        self.na = -1
        self.x_min = None
        self.xres = None
        self.xskew = None
        self.y_max = None
        self.yskew = None
        self.yres = None
        self.x_max = None
        self.y_min = None
        self.x_res = None
        self.y_res = None
        self.pixel_width = None
        # self.meters = None
        self.GeoTransform = None
        self.Projection = None
        self.pushButton_5.clicked.connect(self.select_ROAD_file)
        self.pushButton_6.clicked.connect(self.load_centroids)

        self.pushButton_2.clicked.connect(self.check_path)
        self.pushButton_8.clicked.connect(self.run2)


if __name__ == "__main__":
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app.setStyle('Fusion')
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
