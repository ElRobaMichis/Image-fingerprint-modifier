from tkinterdnd2 import *
from tkinter import Label, filedialog, Button, Frame, ttk, Checkbutton, BooleanVar
from PIL import Image, ImageTk
import threading
import time
import io
import hashlib

class ImageProcessorUI:
    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.modified_img = None
        self.setup_ui()

    def setup_ui(self):
        # Crear la ventana principal usando TkinterDnD
        self.ventana = TkinterDnD.Tk()
        self.ventana.title("Procesador de Imágenes")
        self.ventana.geometry("800x600")

        # Crear un frame principal para organizar los elementos
        self.main_frame = Frame(self.ventana)
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Crear frames para las imágenes y el área de control
        self.images_frame = Frame(self.main_frame)
        self.images_frame.pack(fill='x', pady=(0, 20))

        self.control_frame = Frame(self.main_frame)
        self.control_frame.pack(fill='x')

        self.setup_image_frames()
        self.setup_pipeline_controls()
        self.setup_control_area()
        self.setup_buttons()
        self.setup_drop_area()

    def setup_pipeline_controls(self):
        # Frame para los controles del pipeline
        self.pipeline_frame = Frame(self.main_frame)
        self.pipeline_frame.pack(fill='x', pady=(0, 10))

        # Label para la sección
        Label(self.pipeline_frame, text="Configuración del Pipeline", font=('Helvetica', 10, 'bold')).pack(pady=(0, 5))

        # Variables para los checkboxes
        self.metadata_var = BooleanVar(value=True)
        self.geometric_var = BooleanVar(value=True)
        self.noise_var = BooleanVar(value=True)
        self.compression_var = BooleanVar(value=True)

        # Frame para organizar los checkboxes en una fila
        checkbox_frame = Frame(self.pipeline_frame)
        checkbox_frame.pack()

        # Checkboxes para cada paso del pipeline
        Checkbutton(checkbox_frame, text="Metadatos", variable=self.metadata_var, 
                   command=self.update_pipeline_config).pack(side='left', padx=5)
        Checkbutton(checkbox_frame, text="Transformaciones", variable=self.geometric_var, 
                   command=self.update_pipeline_config).pack(side='left', padx=5)
        Checkbutton(checkbox_frame, text="Ruido", variable=self.noise_var, 
                   command=self.update_pipeline_config).pack(side='left', padx=5)
        Checkbutton(checkbox_frame, text="Compresión", variable=self.compression_var, 
                   command=self.update_pipeline_config).pack(side='left', padx=5)

    def update_pipeline_config(self):
        config = {
            'metadata': self.metadata_var.get(),
            'geometric': self.geometric_var.get(),
            'noise': self.noise_var.get(),
            'compression': self.compression_var.get()
        }
        self.image_processor.set_pipeline_config(config)

    def setup_image_frames(self):
        # Crear frames para cada imagen
        self.original_frame = Frame(self.images_frame)
        self.original_frame.pack(side='left', expand=True, fill='both', padx=(0, 10))
        self.modified_frame = Frame(self.images_frame)
        self.modified_frame.pack(side='right', expand=True, fill='both', padx=(10, 0))

        # Labels para los títulos de las imágenes
        Label(self.original_frame, text="Imagen Original").pack()
        Label(self.modified_frame, text="Imagen Modificada").pack()

        # Labels para las imágenes
        self.preview_original = Label(self.original_frame)
        self.preview_original.pack(pady=5)
        self.preview_modified = Label(self.modified_frame)
        self.preview_modified.pack(pady=5)

        # Labels para los hashes
        self.hash_original = Label(self.original_frame, text="", wraplength=300)
        self.hash_original.pack(pady=5)
        self.hash_modified = Label(self.modified_frame, text="", wraplength=300)
        self.hash_modified.pack(pady=5)

    def setup_control_area(self):
        # Crear la barra de progreso (inicialmente oculta)
        self.progress_bar = ttk.Progressbar(self.control_frame, mode='determinate', length=300)

    def setup_buttons(self):
        # Crear un frame adicional para los botones y centrarlo
        self.buttons_frame = Frame(self.control_frame)
        self.buttons_frame.pack(expand=True, fill='x')

        # Crear un frame contenedor para los botones que los centre
        self.buttons_container = Frame(self.buttons_frame)
        self.buttons_container.pack(expand=True, anchor='center')

        # Crear los botones en el contenedor de botones
        self.boton_modificar = Button(self.buttons_container, text="Modificar imagen", 
                                    state="disabled", command=self.start_modification)
        self.boton_modificar.pack(side='left', padx=5, pady=5)

        self.boton_guardar = Button(self.buttons_container, text="Guardar imagen", 
                                  state="disabled", command=self.guardar_imagen)
        self.boton_guardar.pack(side='left', padx=5, pady=5)

    def setup_drop_area(self):
        # Crear el área de drop
        self.drop_area = Label(self.control_frame, 
                             text="Arrastra y suelta archivos aquí\no haz clic para seleccionar un archivo",
                             width=40, height=5,
                             relief="solid",
                             bg="white",
                             fg="black")
        self.drop_area.pack(pady=10)

        # Registrar los eventos de drag & drop
        self.drop_area.drop_target_register("DND_Files")
        self.drop_area.dnd_bind('<<DropEnter>>', self.drag_enter)
        self.drop_area.dnd_bind('<<DropLeave>>', self.drag_leave)
        self.drop_area.dnd_bind('<<Drop>>', self.drop)
        self.drop_area.bind('<Button-1>', self.on_click)

    def start_modification(self):
        self.boton_modificar.config(state="disabled")
        self.progress_bar.pack(pady=10)
        self.progress_bar['value'] = 0
        
        def update_progress():
            for i in range(101):
                self.progress_bar['value'] = i
                self.ventana.update_idletasks()
                time.sleep(0.02)
            
            self.modificar_imagen()
            self.progress_bar.pack_forget()
            self.boton_modificar.config(state="normal")
        
        thread = threading.Thread(target=update_progress)
        thread.start()

    def modificar_imagen(self):
        original_path = self.drop_area.cget("text").split("\n")[1]
        
        try:
            with Image.open(original_path) as img:
                # Procesar la imagen
                self.modified_img = self.image_processor.process_image(img)
                
                # Crear y mostrar la vista previa
                preview = self.modified_img.copy()
                preview_bio = self.image_processor.resize_image_for_preview(preview)
                modified_photo = ImageTk.PhotoImage(Image.open(preview_bio))
                self.preview_modified.image = modified_photo
                self.preview_modified.config(image=modified_photo)
                self.boton_guardar.config(state="normal")
                
                # Calcular y mostrar hashes
                original_hash = self.image_processor.calculate_image_hash(original_path)
                temp_buffer = io.BytesIO()
                self.modified_img.save(temp_buffer, format='PNG', optimize=True)
                new_hash = hashlib.md5(temp_buffer.getvalue()).hexdigest()
                
                self.hash_original.config(text=f"Hash: {original_hash}")
                self.hash_modified.config(text=f"Hash: {new_hash}\n{'✓ Hashes diferentes' if original_hash != new_hash else '❌ Hashes iguales'}")

        except Exception as e:
            print(f"Error al modificar la imagen: {e}")

    def guardar_imagen(self):
        if self.modified_img:
            try:
                original_path = self.drop_area.cget("text").split("\n")[1]
                new_path = self.image_processor.get_new_filename(original_path)
                saved_path = self.image_processor.save_image(self.modified_img, original_path, new_path)
                self.drop_area.config(text=f"Imagen modificada guardada como:\n{saved_path}")
            except Exception as e:
                print(f"Error al guardar la imagen: {e}")

    def drag_enter(self, event):
        self.drop_area.config(bg="lightblue")

    def drag_leave(self, event):
        self.drop_area.config(bg="white")

    def get_valid_image_extensions(self):
        return ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    def is_valid_image(self, filename):
        return filename.lower().endswith(self.get_valid_image_extensions())

    def drop(self, event):
        self.drop_area.config(bg="white")
        archivo = event.data
        if archivo:
            archivo = archivo.strip('{}')
            if self.is_valid_image(archivo):
                self.drop_area.config(text=f"Imagen recibida:\n{archivo}")
                print(f"Imagen recibida: {archivo}")
                self.boton_modificar.config(state="normal")
                # Limpiar hashes y vista previa modificada
                self.hash_original.config(text="")
                self.hash_modified.config(text="")
                self.boton_guardar.config(state="disabled")
                try:
                    with Image.open(archivo) as img:
                        preview_bio = self.image_processor.resize_image_for_preview(img)
                        photo = ImageTk.PhotoImage(Image.open(preview_bio))
                        self.preview_original.image = photo
                        self.preview_original.config(image=photo)
                        self.preview_modified.config(image='')
                except Exception as e:
                    print(f"Error al cargar la imagen: {e}")
            else:
                self.drop_area.config(text="Por favor, arrastra solo archivos de imagen\n" + 
                                   "Formatos permitidos: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP")
                self.boton_modificar.config(state="disabled")
                self.preview_original.config(image='')
                self.preview_modified.config(image='')
                self.hash_original.config(text="")
                self.hash_modified.config(text="")
                self.boton_guardar.config(state="disabled")

    def on_click(self, event):
        if event.widget == self.drop_area:
            archivo = filedialog.askopenfilename(
                filetypes=[
                    ("Archivos de imagen", 
                     " ".join(f"*{ext}" for ext in self.get_valid_image_extensions())),
                    ("Todos los archivos", "*.*")
                ]
            )
            if archivo:
                if self.is_valid_image(archivo):
                    self.drop_area.config(text=f"Imagen seleccionada:\n{archivo}")
                    print(f"Imagen seleccionada: {archivo}")
                    self.boton_modificar.config(state="normal")
                    # Limpiar hashes y vista previa modificada
                    self.hash_original.config(text="")
                    self.hash_modified.config(text="")
                    self.boton_guardar.config(state="disabled")
                    try:
                        with Image.open(archivo) as img:
                            preview_bio = self.image_processor.resize_image_for_preview(img)
                            photo = ImageTk.PhotoImage(Image.open(preview_bio))
                            self.preview_original.image = photo
                            self.preview_original.config(image=photo)
                            self.preview_modified.config(image='')
                    except Exception as e:
                        print(f"Error al cargar la imagen: {e}")
                else:
                    self.drop_area.config(text="Por favor, selecciona solo archivos de imagen\n" + 
                                       "Formatos permitidos: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP")
                    self.boton_modificar.config(state="disabled")
                    self.preview_original.config(image='')
                    self.preview_modified.config(image='')
                    self.hash_original.config(text="")
                    self.hash_modified.config(text="")
                    self.boton_guardar.config(state="disabled")

    def run(self):
        self.ventana.mainloop() 