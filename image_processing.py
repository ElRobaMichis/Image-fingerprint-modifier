from PIL import Image
import numpy as np
import random
import string
import hashlib
import time
import io
import os

class ImageProcessor:
    def __init__(self):
        self.pipeline_config = {
            'metadata': True,
            'geometric': True,
            'noise': True,
            'compression': True,
            'alpha_variations': True
        }

    def set_pipeline_config(self, config):
        self.pipeline_config.update(config)

    @staticmethod
    def calculate_image_hash(image_path):
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    @staticmethod
    def get_new_filename(original_path):
        directory = os.path.dirname(original_path)
        _, ext = os.path.splitext(original_path)
        
        # Generar nombre: IMG_ + 6 letras + _ + 6 números
        random_letters = ''.join(random.choices(string.ascii_uppercase, k=6))
        random_numbers = ''.join(random.choices(string.digits, k=6))
        new_name = f"IMG_{random_letters}_{random_numbers}{ext}"
        return os.path.join(directory, new_name)

    @staticmethod
    def resize_image_for_preview(img, max_size=(200, 200)):
        img.thumbnail(max_size, Image.Resampling.BILINEAR)
        bio = io.BytesIO()
        img.save(bio, format='PNG', optimize=True)
        bio.seek(0)
        return bio

    def apply_geometric_transformations(self, img):
        if not self.pipeline_config['geometric']:
            return img

        width, height = img.size
        min_dimension = min(width, height)
        
        # Recorte mínimo proporcional al tamaño de la imagen
        max_border_crop = max(1, min(min_dimension // 200, 2))  # Limitar el recorte según el tamaño
        border_crop = random.randint(1, max_border_crop)
        
        # Aplicar recorte y reescalar al tamaño original
        processed = img.crop((border_crop, border_crop, 
                            width - border_crop, height - border_crop))
        return processed.resize(img.size, Image.Resampling.BILINEAR)

    def is_uniform_image(self, img_array, threshold=10):
        """Detecta si una imagen es uniforme (color sólido o degradado sutil)"""
        if len(img_array.shape) == 2:  # Grayscale
            std_dev = np.std(img_array)
            return std_dev < threshold
        else:  # RGB
            std_dev = np.std(img_array, axis=(0, 1))
            return np.all(std_dev < threshold)

    def apply_noise(self, img):
        if not self.pipeline_config['noise']:
            return img

        # Convert to NumPy array for pixel modifications
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        # Detectar si la imagen es pequeña o uniforme
        is_small = total_pixels < 10000  # Menos de 100x100 píxeles
        is_uniform = self.is_uniform_image(img_array)
        
        # Ajustar el porcentaje de píxeles a modificar según el caso
        if is_small or is_uniform:
            # Para imágenes pequeñas o uniformes, usar un porcentaje mayor
            percentage = 0.005 if is_uniform else 0.002  # 0.5% o 0.2%
            num_pixels_to_change = max(20, int(total_pixels * percentage))
        else:
            # Para imágenes normales, mantener el porcentaje bajo
            if total_pixels > 1000000:  # Más de 1 megapíxel
                num_pixels_to_change = min(20, int(total_pixels * 0.00001))  # 0.001% con máximo de 20 píxeles
            else:
                num_pixels_to_change = max(10, int(total_pixels * 0.0001))  # 0.01% con mínimo de 10 píxeles
        
        # Create random indices for pixel modification
        flat_indices = np.random.choice(height * width, num_pixels_to_change, replace=False)
        rows = flat_indices // width
        cols = flat_indices % width
        
        # Asegurar que trabajamos con uint8 desde el principio
        modified = img_array.astype(np.uint8)
        
        if len(img_array.shape) == 2:  # Grayscale
            # Aumentar el rango de ruido para imágenes uniformes
            noise_range = (-5, 6) if is_uniform else (-3, 4)
            noise = np.random.randint(noise_range[0], noise_range[1], num_pixels_to_change, dtype=np.int8)
            modified[rows, cols] = np.clip(
                modified[rows, cols].astype(np.int16) + noise,
                0, 255
            ).astype(np.uint8)
        else:  # RGB
            # Aumentar el rango de transformaciones para imágenes uniformes
            if is_uniform:
                transformaciones = np.array([
                    (-3, 2),  # Rango ampliado para R
                    (-2, 3),  # Rango ampliado para G
                    (-2, 2)   # Rango ampliado para B
                ], dtype=np.int8)
            else:
                transformaciones = np.array([
                    (-2, 1),  # Rango normal para R
                    (-1, 2),  # Rango normal para G
                    (-1, 1)   # Rango normal para B
                ], dtype=np.int8)
            
            channels = np.random.randint(0, 3, num_pixels_to_change)
            min_vals = transformaciones[channels, 0]
            max_vals = transformaciones[channels, 1]
            noise = np.random.randint(min_vals, max_vals + 1, dtype=np.int8)
            
            channel_masks = np.zeros((num_pixels_to_change, 3), dtype=bool)
            channel_masks[np.arange(num_pixels_to_change), channels] = True
            
            modified_pixels = modified[rows, cols].astype(np.int16)
            noise_3d = np.where(channel_masks, noise[:, np.newaxis], 0)
            modified_pixels += noise_3d
            modified[rows, cols] = np.clip(modified_pixels, 0, 255).astype(np.uint8)
        
        return Image.fromarray(modified)

    def apply_lsb_modification(self, img):
        """Aplica cambios ultra-sutiles en el LSB (Least Significant Bit) de algunos píxeles"""
        # Convert to NumPy array for pixel modifications
        arr = np.array(img)
        height, width = arr.shape[:2]
        total_pixels = height * width
        
        # Calcular número de píxeles a modificar (0.001%)
        num_pixels = max(10, int(total_pixels * 0.00001))  # Mínimo 10 píxeles
        
        # Seleccionar píxeles aleatorios
        indices = np.random.choice(height * width, num_pixels, replace=False)
        
        # Modificar LSB
        for idx in indices:
            r = idx // width
            c = idx % width
            if arr.ndim == 2:  # Grayscale
                # Alterna el bit menos significativo
                arr[r, c] ^= 1
            else:  # RGB
                # Elige un canal al azar y modifica su LSB
                chan = np.random.randint(0, 3)
                arr[r, c, chan] ^= 1
        
        return Image.fromarray(arr)

    def apply_alpha_variations(self, img):
        """Añade un canal alfa con variaciones sutiles manteniendo opacidad total visual"""
        # Convertir a RGBA si no lo está ya
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convertir a array para modificar el canal alfa
        arr = np.array(img)
        height, width = arr.shape[:2]
        total_pixels = height * width
        
        # Seleccionar un pequeño porcentaje de píxeles para modificar (0.1%)
        num_pixels = max(20, int(total_pixels * 0.001))
        
        # Seleccionar píxeles aleatorios
        indices = np.random.choice(height * width, num_pixels, replace=False)
        rows = indices // width
        cols = indices % width
        
        # Generar valores aleatorios en el rango [250, 254]
        # Nota: Mantenemos 255 como máximo para la mayoría de píxeles
        alpha_values = np.random.randint(250, 255, num_pixels, dtype=np.uint8)
        
        # Modificar el canal alfa (índice 3) en las posiciones seleccionadas
        arr[rows, cols, 3] = alpha_values
        
        return Image.fromarray(arr)

    def process_image(self, img):
        # Convert to RGB only if necessary
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Aplicar transformaciones geométricas
        img = self.apply_geometric_transformations(img)
        
        # Aplicar ruido
        img = self.apply_noise(img)
        
        # Aplicar modificación LSB
        img = self.apply_lsb_modification(img)
        
        # Aplicar variaciones en canal alfa si se guardará como PNG
        if self.pipeline_config.get('alpha_variations', True):
            img = self.apply_alpha_variations(img)
        
        # Añadir metadatos si está habilitado
        if self.pipeline_config['metadata']:
            img, random_comment = self.add_metadata(img)
            if img.format == 'PNG':
                img = self.add_png_metadata(img, random_comment)
        
        return img

    def add_metadata(self, img):
        """Añade metadatos a la imagen y retorna la imagen modificada junto con el comentario generado"""
        random_comment = ''.join(random.choices(string.ascii_letters + string.digits, k=64))
        random_software_version = f'{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,999)}'
        random_processing_software = f'CustomImageProcessor-{random.randint(1000, 9999)}'
        
        # Crear un comentario de usuario largo para EXIF
        long_comment = ''.join(random.choices(string.ascii_letters + string.digits, k=2048))
        
        # Crear diccionario de metadatos
        metadata = {
            'Software': random_processing_software,
            'ProcessingSoftware': f'CIP-{random_software_version}',
            'DateTime': time.strftime('%Y:%m:%d %H:%M:%S'),
            'ModifyDate': time.strftime('%Y:%m:%d %H:%M:%S'),
            'UserComment': f'[{long_comment}]',  # Comentario largo
            'Artist': 'BorderFrameApp',
            'Copyright': f'© {time.strftime("%Y")} BorderFrameApp',
            'ImageUniqueID': hashlib.md5(random_comment.encode()).hexdigest(),
            'ProcessingParams': 'geometric,noise,metadata'
        }
        
        # Actualizar metadatos existentes
        if hasattr(img, 'info'):
            img.info.update(metadata)
        
        return img, random_comment

    def add_png_metadata(self, img, random_comment):
        """Añade metadatos específicos para imágenes PNG usando chunks tEXt y otros"""
        if not isinstance(img, Image.Image):
            return img

        # Detectar si la imagen es pequeña o uniforme
        img_array = np.array(img)
        is_small = img_array.size < 10000
        is_uniform = self.is_uniform_image(img_array)
        
        # Ajustar el tamaño de los chunks según el caso
        chunk_size = 1024 if (is_small or is_uniform) else 512  # Duplicar el tamaño para imágenes pequeñas/uniformes

        # Crear un nuevo objeto PngInfo
        meta = Image.PngInfo()
        
        # Añadir chunks tEXt básicos
        meta.add_text('Software', img.info.get('Software', ''))
        meta.add_text('ProcessingSoftware', img.info.get('ProcessingSoftware', ''))
        meta.add_text('DateTime', img.info.get('DateTime', ''))
        meta.add_text('ModifyDate', img.info.get('ModifyDate', ''))
        meta.add_text('UserComment', random_comment)
        meta.add_text('Artist', img.info.get('Artist', ''))
        meta.add_text('Copyright', img.info.get('Copyright', ''))
        meta.add_text('ImageUniqueID', img.info.get('ImageUniqueID', ''))
        meta.add_text('ProcessingParams', img.info.get('ProcessingParams', ''))

        # Añadir chunks personalizados con datos aleatorios
        custom_chunks = {
            'meTa': ''.join(random.choices(string.ascii_letters + string.digits, k=chunk_size)),
            'junK': ''.join(random.choices(string.ascii_letters + string.digits, k=chunk_size)),
            'ranD': ''.join(random.choices(string.ascii_letters + string.digits, k=chunk_size)),
            'datA': ''.join(random.choices(string.ascii_letters + string.digits, k=chunk_size))
        }
        
        # Añadir los chunks personalizados
        for key, value in custom_chunks.items():
            meta.add_text(key, value)

        # Añadir chunks tEXt adicionales con información aleatoria
        random_chunks = {
            'CustomData': ''.join(random.choices(string.ascii_letters + string.digits, k=32)),
            'ProcessingTime': time.strftime('%Y%m%d%H%M%S'),
            'RandomSeed': str(random.randint(1000000, 9999999)),
            'TransformationID': hashlib.sha256(random_comment.encode()).hexdigest()[:16],
            'NoiseLevel': f"{random.uniform(0.001, 0.002):.6f}",
            'GeometricParams': f"rotation={random.uniform(-0.15, 0.15):.4f},crop={random.randint(1, 2)}",
            'BorderFrameVersion': f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,99)}",
            'ProcessingFlags': ','.join(random.sample(['noise', 'geometric', 'metadata', 'compression', 'transform'], 3))
        }

        # Añadir los chunks adicionales
        for key, value in random_chunks.items():
            meta.add_text(key, value)
        
        # Guardar temporalmente la imagen con los metadatos
        temp_buffer = io.BytesIO()
        img.save(temp_buffer, format='PNG', pnginfo=meta, optimize=True)
        temp_buffer.seek(0)
        
        # Recargar la imagen con los metadatos
        return Image.open(temp_buffer)

    def verify_metadata(self, img):
        """Verifica que los metadatos se hayan guardado correctamente"""
        required_fields = ['Software', 'ProcessingSoftware', 'DateTime', 'UserComment']
        
        if not hasattr(img, 'info'):
            return False, "La imagen no tiene atributo info"
            
        missing_fields = [field for field in required_fields if field not in img.info]
        
        if missing_fields:
            return False, f"Faltan los siguientes campos de metadatos: {', '.join(missing_fields)}"
        
        # Verificar campos adicionales para PNG
        if img.format == 'PNG':
            png_fields = ['CustomData', 'ProcessingTime', 'RandomSeed', 'TransformationID']
            png_missing = [field for field in png_fields if field not in img.info]
            if png_missing:
                return True, "Metadatos básicos presentes, algunos campos PNG opcionales faltantes"
            
        return True, "Todos los metadatos requeridos están presentes"

    def save_image(self, img, original_path, new_path):
        """Guarda la imagen con sus metadatos y variaciones en la compresión"""
        # Calcular el hash original
        original_hash = self.calculate_image_hash(original_path)
        
        # Determinar el formato de salida
        format_out = 'PNG' if new_path.lower().endswith('.png') else 'JPEG'
        
        if format_out == 'PNG':
            # Preparar los metadatos básicos y chunks personalizados
            chunks_data = []
            
            # Metadatos básicos
            meta = Image.PngInfo()
            for k, v in img.info.items():
                if isinstance(v, str):
                    meta.add_text(k, v)
                    chunks_data.append(('tEXt', k, v))
            
            # Generar chunks personalizados adicionales con datos aleatorios
            custom_chunks = {
                'meTa': ''.join(random.choices(string.ascii_letters + string.digits, k=512)),
                'junK': ''.join(random.choices(string.ascii_letters + string.digits, k=512)),
                'ranD': ''.join(random.choices(string.ascii_letters + string.digits, k=512)),
                'datA': ''.join(random.choices(string.ascii_letters + string.digits, k=512))
            }
            
            for key, value in custom_chunks.items():
                meta.add_text(key, value)
                chunks_data.append(('tEXt', key, value))
            
            # Generar un perfil ICC aleatorio
            icc_profile = bytes([
                0, 0, 2, 0,  # Tamaño del perfil
                *random.choices(range(256), k=508),  # Datos aleatorios
                0, 0, 0, 0   # Checksum
            ])
            
            if self.pipeline_config['compression']:
                # Primera pasada: Guardar con un orden de chunks y compresión
                temp_buffer1 = io.BytesIO()
                random.shuffle(chunks_data)  # Reordenar chunks aleatoriamente
                meta1 = Image.PngInfo()
                for chunk_type, key, value in chunks_data:
                    meta1.add_text(key, value)
                
                img.save(temp_buffer1, format='PNG',
                        pnginfo=meta1,
                        icc_profile=icc_profile,
                        optimize=True,
                        compress_level=random.randint(6, 9))
                
                # Segunda pasada: Recargar y guardar con diferente orden y compresión
                temp_buffer1.seek(0)
                temp_img = Image.open(temp_buffer1)
                temp_buffer2 = io.BytesIO()
                
                random.shuffle(chunks_data)  # Reordenar chunks nuevamente
                meta2 = Image.PngInfo()
                for chunk_type, key, value in chunks_data:
                    meta2.add_text(key, value)
                
                temp_img.save(temp_buffer2, format='PNG',
                            pnginfo=meta2,
                            icc_profile=icc_profile,
                            optimize=True,
                            compress_level=random.randint(6, 9))
                
                # Tercera pasada: Guardar final con otro orden y compresión
                temp_buffer2.seek(0)
                temp_img = Image.open(temp_buffer2)
                
                random.shuffle(chunks_data)  # Reordenar chunks una vez más
                meta3 = Image.PngInfo()
                for chunk_type, key, value in chunks_data:
                    meta3.add_text(key, value)
                
                temp_img.save(new_path, format='PNG',
                            pnginfo=meta3,
                            icc_profile=icc_profile,
                            optimize=True,
                            compress_level=random.randint(6, 9))
            else:
                # Sin compresión, pero aún reordenamos los chunks
                random.shuffle(chunks_data)
                meta_final = Image.PngInfo()
                for chunk_type, key, value in chunks_data:
                    meta_final.add_text(key, value)
                
                img.save(new_path, format='PNG',
                        pnginfo=meta_final,
                        icc_profile=icc_profile,
                        optimize=False,
                        compress_level=0)
        else:
            # Para JPEG, convertir a RGB ya que JPEG no soporta alfa
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            # Variar la calidad y los parámetros de compresión
            if self.pipeline_config['compression']:
                # Variar la calidad en un rango alto
                quality = random.randint(95, 97)
                
                # Generar subsampling aleatorio
                subsampling = random.choice(['4:4:4', '4:2:2', '4:2:0'])
                
                # Crear un buffer temporal
                temp_buffer = io.BytesIO()
                img.save(temp_buffer, format='JPEG',
                        quality=quality,
                        subsampling=subsampling,
                        icc_profile=icc_profile,
                        optimize=True)
                
                # Recargar y volver a guardar con diferentes parámetros
                temp_buffer.seek(0)
                temp_img = Image.open(temp_buffer)
                
                # Segunda pasada con calidad ligeramente diferente
                quality_2 = quality + random.choice([-1, 1])
                temp_img.save(new_path, format='JPEG',
                            quality=quality_2,
                            subsampling=random.choice(['4:4:4', '4:2:2', '4:2:0']),
                            icc_profile=icc_profile,
                            optimize=True)
            else:
                # Sin compresión, usar calidad máxima
                img.save(new_path, format='JPEG',
                        quality=100,
                        subsampling='4:4:4',
                        icc_profile=icc_profile,
                        optimize=False)
        
        # Verificar que los metadatos se guardaron correctamente
        with Image.open(new_path) as saved_img:
            success, message = self.verify_metadata(saved_img)
            if not success:
                print(f"Advertencia: {message}")
        
        # Verificar el hash de la imagen guardada
        new_hash = self.calculate_image_hash(new_path)
        
        # Si los hashes son iguales, aplicar ruido adicional y guardar de nuevo
        if new_hash == original_hash:
            print("Hash coincidente detectado. Aplicando ruido adicional...")
            img = self.apply_noise(img)  # Aplicar ruido adicional
            
            # Guardar la imagen con ruido adicional
            if format_out == 'PNG':
                # Reordenar chunks una vez más
                random.shuffle(chunks_data)
                meta_retry = Image.PngInfo()
                for chunk_type, key, value in chunks_data:
                    meta_retry.add_text(key, value)
                
                img.save(new_path, format='PNG',
                        pnginfo=meta_retry,
                        icc_profile=icc_profile,
                        optimize=True,
                        compress_level=random.randint(6, 9))
            else:
                img.save(new_path, format='JPEG',
                        quality=random.randint(95, 97),
                        subsampling=random.choice(['4:4:4', '4:2:2', '4:2:0']),
                        icc_profile=icc_profile,
                        optimize=True)
            
            # Verificar el nuevo hash
            final_hash = self.calculate_image_hash(new_path)
            if final_hash == original_hash:
                print("Advertencia: El hash sigue siendo igual después del ruido adicional")
        
        return new_path 