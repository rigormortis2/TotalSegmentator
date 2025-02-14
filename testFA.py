import sys, os
import numpy as np, pydicom
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
                             QGraphicsPolygonItem, QFileDialog, QPushButton, QVBoxLayout, QWidget, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QPolygonF, QPen, QPainter, QColor
from PyQt5.QtCore import Qt, QPointF, QTimer

# --- Yardımcı Fonksiyonlar ---
def distance_between_points(p1, p2):
    return ((p1.x() - p2.x()) ** 2 + (p1.y() - p2.y()) ** 2) ** 0.5

def distance_point_to_segment(p, a, b):
    ab = b - a
    ap = p - a
    ab_len2 = ab.x()**2 + ab.y()**2
    if ab_len2 == 0:
        return distance_between_points(p, a)
    t = (ap.x()*ab.x() + ap.y()*ab.y()) / ab_len2
    if t < 0.0:
        return distance_between_points(p, a)
    elif t > 1.0:
        return distance_between_points(p, b)
    projection = QPointF(a.x() + t * ab.x(), a.y() + t * ab.y())
    return distance_between_points(p, projection)

# --- Sürüklenebilir Poligon Öğesi ---
class DraggablePolygon(QGraphicsPolygonItem):
    def __init__(self, polygon, slice_index, is_user, move_callback, edge_edit_callback, parent=None):
        super(DraggablePolygon, self).__init__(polygon, parent)
        self.slice_index = slice_index
        self.is_user = is_user  # True: kullanıcı tarafından çizilmiş, False: interpolasyon sonucu
        self.move_callback = move_callback
        self.edge_edit_callback = edge_edit_callback
        self.setFlags(QGraphicsPolygonItem.ItemIsMovable | QGraphicsPolygonItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)
        self.hovered_edge_start_index = None
        self.hovered_edge_end_index = None

    def hoverMoveEvent(self, event):
        pos = event.pos()  # yerel koordinatlar
        poly = self.polygon()
        threshold = 5.0  # piksel cinsinden yakınlık eşiği
        self.hovered_edge_start_index = None
        self.hovered_edge_end_index = None
        n = poly.count()
        for i in range(n):
            j = (i + 1) % n
            p1 = poly.at(i)
            p2 = poly.at(j)
            d = distance_point_to_segment(pos, p1, p2)
            if d < threshold:
                self.hovered_edge_start_index = i
                self.hovered_edge_end_index = j
                break
        if self.hovered_edge_start_index is not None:
            self.setCursor(Qt.IBeamCursor)  # kenar düzenleme imleci
        else:
            self.setCursor(Qt.OpenHandCursor)
        super(DraggablePolygon, self).hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.hovered_edge_start_index is not None:
            scene_pos = self.mapToScene(event.pos())
            if self.edge_edit_callback:
                self.edge_edit_callback(self.slice_index, QPolygonF(self.polygon()),
                                        self.hovered_edge_start_index, self.hovered_edge_end_index, scene_pos)
            event.accept()
            return
        self.setCursor(Qt.ClosedHandCursor)
        super(DraggablePolygon, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        new_poly = QPolygonF([self.mapToScene(pt) for pt in self.polygon()])
        if self.move_callback:
            QTimer.singleShot(0, lambda: self.move_callback(self.slice_index, new_poly))
        super(DraggablePolygon, self).mouseReleaseEvent(event)

# --- MR Viewer ---
class MRViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(MRViewer, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.slices = []          # MR slice’lar (numpy array)
        self.current_slice_index = 0
        self.pixmap_item = None

        # Free-hand çizim için
        self.drawing_mode = False
        self.drawing = False
        self.current_polygon_points = []  # geçici çizim noktaları

        # Kenar düzenleme (edge editing) için değişkenler
        self.edge_edit_mode = False
        self.editing_polygon = None      # Düzenlenecek orijinal poligon (QPolygonF)
        self.editing_slice_index = None
        self.editing_start_index = None  # Düzenlenecek kenarın başlangıç indeksi
        self.editing_end_index = None    # Kenarın diğer ucu
        self.editing_points = QPolygonF()  # Kullanıcının çizdiği yeni kenar (henüz tamamlanmamış)

        # Kullanıcı (elle çizilmiş) ve interpolasyon poligonları
        self.user_polygons = {}
        self.interpolated_polygons = {}

        self.setRenderHint(QPainter.Antialiasing)

    def loadMRFiles(self, filepaths):
        filepaths = sorted(filepaths)
        slices = []
        for fp in filepaths:
            ds = pydicom.dcmread(fp)
            img = ds.pixel_array.astype(np.float32)
            img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
            slices.append(img.astype(np.uint8))
        self.slices = slices
        self.current_slice_index = 0
        self.user_polygons.clear()
        self.interpolated_polygons.clear()
        self.updateSlice()

    def updateSlice(self):
        if not self.slices:
            return
        img = self.slices[self.current_slice_index]
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

        if self.edge_edit_mode and self.editing_polygon is not None:
            merged_poly = self.getMergedPolygon()
            # Mavi kenarın kalınlığı 1 olarak ayarlandı.
            self.scene.addPolygon(merged_poly, QPen(QColor("blue"), 1, Qt.DashLine))
        else:
            if self.current_slice_index in self.user_polygons:
                poly = self.user_polygons[self.current_slice_index]
                item = DraggablePolygon(poly, self.current_slice_index, True,
                                        self.polygonMoved, self.startEdgeEditing)
                item.setPen(QPen(Qt.red, 1))
                self.scene.addItem(item)
            elif self.current_slice_index in self.interpolated_polygons:
                poly = self.interpolated_polygons[self.current_slice_index]
                item = DraggablePolygon(poly, self.current_slice_index, False,
                                        self.polygonMoved, self.startEdgeEditing)
                item.setPen(QPen(Qt.green, 1, Qt.DashLine))
                self.scene.addItem(item)
            if self.drawing_mode and self.current_polygon_points:
                temp_poly = QPolygonF(self.current_polygon_points)
                self.scene.addPolygon(temp_poly, QPen(Qt.blue, 1))

    def resizeEvent(self, event):
        super(MRViewer, self).resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.pixmap_item.boundingRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if not self.slices:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.current_slice_index = max(0, self.current_slice_index - 1)
        else:
            self.current_slice_index = min(len(self.slices) - 1, self.current_slice_index + 1)
        self.current_polygon_points = []
        self.updateSlice()

    def mousePressEvent(self, event):
        if self.edge_edit_mode:
            if event.button() == Qt.RightButton:
                # Edge editing iptal ediliyor
                self.edge_edit_mode = False
                self.editing_polygon = None
                self.editing_points = QPolygonF()
                self.editing_start_index = None
                self.editing_end_index = None
                self.updateSlice()
            return
        if self.drawing_mode and event.button() == Qt.LeftButton:
            self.drawing = True
            self.current_polygon_points = []
            pos = self.mapToScene(event.pos())
            self.current_polygon_points.append(pos)
            self.updateSlice()
        else:
            super(MRViewer, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.edge_edit_mode:
            pos = self.mapToScene(event.pos())
            if event.buttons() & Qt.LeftButton:
                self.editing_points.append(pos)
                self.updateSlice()
            return
        elif self.drawing_mode and self.drawing:
            pos = self.mapToScene(event.pos())
            self.current_polygon_points.append(pos)
            self.updateSlice()
        else:
            super(MRViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.edge_edit_mode and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            end_point = self.editing_polygon.at(self.editing_end_index)
            if distance_between_points(pos, end_point) < 10.0:
                # Son nokta snap ediliyor
                self.editing_points.append(end_point)
                merged = self.getMergedPolygon()
                self.user_polygons[self.editing_slice_index] = merged
                self.edge_edit_mode = False
                self.editing_polygon = None
                self.editing_points = QPolygonF()
                self.editing_start_index = None
                self.editing_end_index = None
                self.reinterpolateAll()
            else:
                QMessageBox.information(self, "Bilgi", "Düzenleme iptal edildi: Son nokta yeterince yakın değil.")
                self.edge_edit_mode = False
                self.editing_polygon = None
                self.editing_points = QPolygonF()
                self.editing_start_index = None
                self.editing_end_index = None
            self.updateSlice()
            return
        elif self.drawing_mode and self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            if len(self.current_polygon_points) >= 3:
                self.user_polygons[self.current_slice_index] = QPolygonF(self.current_polygon_points)
                self.reinterpolateAll()
            self.drawing_mode = False
            self.current_polygon_points = []
            self.updateSlice()
            return
        else:
            super(MRViewer, self).mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (self.current_slice_index in self.user_polygons) or (self.current_slice_index in self.interpolated_polygons):
            if self.current_slice_index in self.user_polygons:
                del self.user_polygons[self.current_slice_index]
            if self.current_slice_index in self.interpolated_polygons:
                del self.interpolated_polygons[self.current_slice_index]
            self.drawing_mode = True
            self.current_polygon_points = []
            self.updateSlice()
        else:
            super(MRViewer, self).mouseDoubleClickEvent(event)

    def startDrawing(self):
        self.drawing_mode = True
        self.current_polygon_points = []

    def resamplePolygon(self, poly, numPoints):
        if poly.count() < 2:
            return poly
        points = [poly.at(i) for i in range(poly.count())]
        distances = []
        for i in range(1, len(points)):
            dx = points[i].x() - points[i - 1].x()
            dy = points[i].y() - points[i - 1].y()
            distances.append((dx * dx + dy * dy) ** 0.5)
        cum_length = [0]
        for d in distances:
            cum_length.append(cum_length[-1] + d)
        total_length = cum_length[-1]
        if total_length == 0:
            return poly
        interval = total_length / (numPoints - 1)
        new_points = []
        target = 0
        j = 0
        for i in range(numPoints):
            if i == numPoints - 1:
                new_points.append(points[-1])
                break
            while j < len(cum_length) - 1 and cum_length[j + 1] < target:
                j += 1
            if cum_length[j] == target:
                new_points.append(points[j])
            else:
                segment_length = cum_length[j + 1] - cum_length[j]
                if segment_length == 0:
                    new_points.append(points[j])
                else:
                    t_val = (target - cum_length[j]) / segment_length
                    x = points[j].x() + t_val * (points[j + 1].x() - points[j].x())
                    y = points[j].y() + t_val * (points[j + 1].y() - points[j].y())
                    new_points.append(QPointF(x, y))
            target += interval
        new_poly = QPolygonF()
        for pt in new_points:
            new_poly.append(pt)
        return new_poly

    def reinterpolateAll(self):
        self.interpolated_polygons.clear()
        control_idxs = sorted(self.user_polygons.keys())
        if len(control_idxs) < 2:
            return
        for idx in range(len(control_idxs) - 1):
            start = control_idxs[idx]
            end = control_idxs[idx + 1]
            poly_start = self.user_polygons[start]
            poly_end = self.user_polygons[end]
            n_target = max(poly_start.count(), poly_end.count())
            poly_start_rs = self.resamplePolygon(poly_start, n_target)
            poly_end_rs = self.resamplePolygon(poly_end, n_target)
            for s in range(start + 1, end):
                t = (s - start) / (end - start)
                interp_poly = QPolygonF()
                for j in range(n_target):
                    p1 = poly_start_rs.at(j)
                    p2 = poly_end_rs.at(j)
                    x = p1.x() + t * (p2.x() - p1.x())
                    y = p1.y() + t * (p2.y() - p1.y())
                    interp_poly.append(QPointF(x, y))
                self.interpolated_polygons[s] = interp_poly

    def polygonMoved(self, slice_index, new_polygon):
        self.user_polygons[slice_index] = new_polygon
        if slice_index in self.interpolated_polygons:
            del self.interpolated_polygons[slice_index]
        self.reinterpolateAll()
        self.updateSlice()

    def interpolatePolygons(self):
        if len(self.user_polygons) < 2:
            QMessageBox.warning(self, "Uyarı",
                                "Lütfen önce en az iki farklı slice’da (örneğin 10 ve 30) elle prostat çizimi yapınız.")
            return
        self.reinterpolateAll()
        self.updateSlice()

    def getMergedPolygon(self):
        """
        Kullanıcının edge-editing sırasında çizdiği yeni kenarın kapalı olup olmadığını kontrol eder.
        - Eğer yeni kenar kapalı (ilk ve son nokta yakın) ise, kullanıcının çizdiği polygon kabul edilir.
        - Aksi halde, orijinal poligonun düzenlenecek arc'ı silinip yerine yeni kenar eklenir.
        """
        if self.editing_polygon is None:
            return QPolygonF()
        # Eğer kullanıcının çizdiği yeni kenar kapalıysa (yeterli nokta varsa)
        if self.editing_points.count() >= 3 and distance_between_points(self.editing_points.first(),
                                                                         self.editing_points.last()) < 10.0:
            return self.editing_points
        else:
            return self.mergePolygonWithCenter(self.editing_polygon,
                                               self.editing_start_index,
                                               self.editing_end_index,
                                               self.editing_points)

    def mergePolygonWithCenter(self, orig_poly, i, j, new_edge):
        """
        Orijinal poligon üzerinden, i ile j arasındaki arc'ın yerine,
        kullanıcının çizdiği new_edge eklenir. İki alternatif (i->j veya j->i) arasından,
        merkezden (ortalama) daha dışta kalan arc kaldırılır.
        """
        n = orig_poly.count()
        vertices = [orig_poly.at(k) for k in range(n)]
        # Poligon merkezi
        cx = sum(pt.x() for pt in vertices) / n
        cy = sum(pt.y() for pt in vertices) / n
        center = QPointF(cx, cy)
        # Arc1: i'den j'ye (orijinal sıra)
        arc1 = []
        idx = i
        while True:
            arc1.append(vertices[idx])
            if idx == j:
                break
            idx = (idx + 1) % n
        # Arc2: j'den i'ye
        arc2 = []
        idx = j
        while True:
            arc2.append(vertices[idx])
            if idx == i:
                break
            idx = (idx + 1) % n
        avg1 = sum(distance_between_points(pt, center) for pt in arc1) / len(arc1)
        avg2 = sum(distance_between_points(pt, center) for pt in arc2) / len(arc2)
        # new_edge'in ilk ve son noktaları orijinalle uyumlu kabul ediliyor.
        if avg1 > avg2:
            # Arc1 dış kısım: onu kaldırıp yerine new_edge koy.
            new_poly = QPolygonF()
            new_poly.append(vertices[i])
            for k in range(1, new_edge.count()):
                new_poly.append(new_edge.at(k))
            for k in range(1, len(arc2)):
                new_poly.append(arc2[k])
            return new_poly
        else:
            # Arc2 dış kısım: onu kaldırıp yerine new_edge (ters yönde) koy.
            new_poly = QPolygonF()
            new_poly.append(vertices[j])
            for k in range(1, new_edge.count()):
                new_poly.append(new_edge.at(k))
            for k in range(1, len(arc1)):
                new_poly.append(arc1[k])
            return new_poly

    def startEdgeEditing(self, slice_index, polygon, start_index, end_index, click_scene_pos):
        """
        DraggablePolygon'dan tetiklenen callback.
        Tespit edilen kenar (start_index, end_index) için edge editing moduna geçilir.
        """
        if (slice_index not in self.user_polygons) and (slice_index not in self.interpolated_polygons):
            return
        if slice_index in self.user_polygons:
            orig_poly = self.user_polygons[slice_index]
        else:
            orig_poly = self.interpolated_polygons[slice_index]
        self.edge_edit_mode = True
        self.editing_slice_index = slice_index
        self.editing_polygon = QPolygonF(orig_poly)
        self.editing_start_index = start_index
        self.editing_end_index = end_index
        self.editing_points = QPolygonF()
        # Başlangıç noktası olarak orijinal kenarın başlangıcı eklenir.
        self.editing_points.append(orig_poly.at(start_index))
        self.updateSlice()

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.viewer = MRViewer(self)

        self.btn_load = QPushButton("MR Dosyalarını Yükle")
        self.btn_draw = QPushButton("Prostat Çizimi Başlat")
        self.btn_interpolate = QPushButton("Interpolasyonu Uygula")

        self.btn_load.clicked.connect(self.loadFiles)
        self.btn_draw.clicked.connect(self.viewer.startDrawing)
        self.btn_interpolate.clicked.connect(self.viewer.interpolatePolygons)

        layout = QVBoxLayout()
        layout.addWidget(self.viewer)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_draw)
        layout.addWidget(self.btn_interpolate)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setWindowTitle("MR Görüntü & Prostat Bölgesi Çizimi")
        self.resize(800, 600)

    def loadFiles(self):
        # Test için klasör yolu (uygulamanıza göre düzenleyin):
        folder_path = r"C:\Users\Fatih\Desktop\testMR"  # MR görüntülerinin bulunduğu klasör yolu
        files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if f.lower().endswith(".dcm")]
        if files:
            self.viewer.loadMRFiles(files)
        # Alternatif olarak dosya seçici de kullanılabilir:
        # options = QFileDialog.Options()
        # files, _ = QFileDialog.getOpenFileNames(self, "MR Dosyalarını Seçin", "",
        #                                         "DICOM Files (*.dcm);;All Files (*)", options=options)
        # if files:
        #     self.viewer.loadMRFiles(files)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())