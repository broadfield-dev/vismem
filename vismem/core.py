import numpy as np
from PIL import Image
import io
import struct
import math
import os

class SectorMemory:
    """
    RGB-Depth Memory Architecture v1.0
    -----------------------------------
    Channel 0 (Red):   Vectors (Fixed 384 bytes per entry)
    Channel 1 (Green): Text Heap (Variable stream, append-only)
    Channel 2 (Blue):  Index/Metadata (12 bytes per entry: Vec_Ptr, Txt_Ptr, Txt_Len)
    """
    def __init__(self, name, width=128, height=128, embedding_dim=384):
        self.name = name
        self.width = width
        self.height = height
        self.dim = embedding_dim
        self.filepath = f"{name}.png"
        
        # Metadata structure: Vec_Ptr (4B) + Txt_Ptr (4B) + Txt_Len (4B)
        self.META_SIZE = 12
        
        # Compatibility attribute (legacy apps expect this)
        self.deleted_slots = set() 
        
        if os.path.exists(self.filepath):
            self._load_from_disk()
        else:
            self._init_grid()

    def _init_grid(self):
        """Creates a fresh blank memory grid."""
        self.grid = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.count = 0       
        self.cursor_green = 0 
        self._save()

    def _save(self):
        """Persists the numpy array to a PNG image."""
        Image.fromarray(self.grid, 'RGB').save(self.filepath)

    def _load_from_disk(self):
        """Loads PNG and scans Blue Channel to recover state."""
        try:
            img = Image.open(self.filepath).convert('RGB')
            self.width, self.height = img.size
            self.grid = np.array(img, dtype=np.uint8)
            
            # Recover State
            self.count = 0
            self.cursor_green = 0
            
            # Calculate max possible entries to prevent overrun
            max_entries = (self.width * self.height) // self.META_SIZE
            
            # Scan Blue Channel (Index)
            for i in range(max_entries):
                # Read 12 bytes
                meta = self._read_bytes(2, i * self.META_SIZE, self.META_SIZE)
                vec_ptr, txt_ptr, txt_len = struct.unpack('>III', bytes(meta))
                
                # Check for null entry (End of Data)
                if vec_ptr == 0 and txt_ptr == 0 and txt_len == 0:
                    break
                
                # Track heap usage
                end_of_text = txt_ptr + txt_len
                if end_of_text > self.cursor_green:
                    self.cursor_green = end_of_text
                
                self.count += 1
                
            print(f"[{self.name}] RGB-Depth Loaded. {self.width}x{self.height}. Rules: {self.count}, TextHeap: {self.cursor_green} bytes.")
            
        except Exception as e:
            print(f"[{self.name}] Load Error: {e}. Re-initializing.")
            self._init_grid()

    def _ensure_capacity(self, needed_vec_bytes, needed_txt_bytes, needed_meta_bytes):
        """Checks if any channel needs more space."""
        total_pixels = self.width * self.height
        
        if needed_vec_bytes > total_pixels or \
           needed_txt_bytes > total_pixels or \
           needed_meta_bytes > total_pixels:
            self._expand()

    def _expand(self):
        """Resizes the grid and reflows the linear data streams."""
        new_w = self.width + 128
        new_h = self.height + 128
        print(f"[{self.name}] Expanding RGB Grid to {new_w}x{new_h}...")
        
        new_grid = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        for ch in range(3):
            old_flat = self.grid[:, :, ch].reshape(-1)
            
            # Determine how much data is actually valid to copy
            if ch == 0: limit = self.count * self.dim      # Red (Vectors)
            elif ch == 1: limit = self.cursor_green        # Green (Text)
            else: limit = self.count * self.META_SIZE      # Blue (Index)
            
            # Extract valid data
            data = old_flat[:limit]
            
            # Write to new grid flattened view
            target_flat = new_grid[:, :, ch].reshape(-1)
            target_flat[:len(data)] = data

        self.width = new_w
        self.height = new_h
        self.grid = new_grid
        self._save()

    def _write_bytes(self, channel, start_idx, data):
        """Writes linear bytes to a specific 2D channel."""
        if isinstance(data, bytes): data = list(data)
        
        curr = start_idx
        for b in data:
            r = curr // self.width
            c = curr % self.width
            self.grid[r, c, channel] = b
            curr += 1

    def _read_bytes(self, channel, start_idx, length):
        """Reads linear bytes from a specific 2D channel."""
        data = []
        curr = start_idx
        
        # Optimization: Pre-calculate end
        end_idx = start_idx + length
        
        # While loop is safer for bounds
        while curr < end_idx:
            r = curr // self.width
            c = curr % self.width
            data.append(self.grid[r, c, channel])
            curr += 1
        return data

    def write_entry(self, text: str, embedding: list, update_id: int = -1):
        """
        Writes a new entry or updates an existing one (Swap/Append).
        """
        text_bytes = text.encode('utf-8')
        t_len = len(text_bytes)
        
        # Quantize Float32 -> Uint8
        q_vec = [int(min(max((val + 1.0) / 2.0 * 255, 0), 255)) for val in embedding]
        
        # Determine ID
        if update_id != -1:
            target_id = update_id
        else:
            target_id = self.count

        # Calculate Addresses
        vec_addr = target_id * self.dim         # Red
        meta_addr = target_id * self.META_SIZE  # Blue
        txt_addr = self.cursor_green            # Green (Always append)
        
        # Expand if needed
        self._ensure_capacity(
            vec_addr + self.dim,
            txt_addr + t_len,
            meta_addr + self.META_SIZE
        )

        # 1. Write Vector (Red)
        self._write_bytes(0, vec_addr, q_vec)
        
        # 2. Write Text (Green)
        self._write_bytes(1, txt_addr, text_bytes)
        self.cursor_green += t_len
        
        # 3. Write Metadata (Blue) -> [Vec_Addr, Txt_Addr, Txt_Len]
        meta = struct.pack('>III', vec_addr, txt_addr, t_len)
        self._write_bytes(2, meta_addr, meta)

        if update_id == -1:
            self.count += 1
            
        self._save()
        return target_id

    def search(self, query_embedding, top_k=3):
        """
        Semantic search using the Red Channel vectors.
        """
        if self.count == 0: return []
        
        results = []
        
        # Scan all active rules
        for i in range(self.count):
            # 1. Read Metadata (Blue)
            meta = self._read_bytes(2, i * self.META_SIZE, self.META_SIZE)
            vec_ptr, txt_ptr, txt_len = struct.unpack('>III', bytes(meta))
            
            # 2. Read Vector (Red)
            vec_bytes = self._read_bytes(0, vec_ptr, self.dim)
            
            # 3. Dequantize
            vec = [(float(b) / 255.0 * 2.0) - 1.0 for b in vec_bytes]
            
            # 4. Dot Product
            score = np.dot(vec, query_embedding)
            
            results.append({
                "id": i, 
                "score": score, 
                "txt_ptr": txt_ptr, 
                "txt_len": txt_len
            })
            
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 5. Fetch Text (Green)
        final = []
        for res in results[:top_k]:
            txt_bytes = self._read_bytes(1, res['txt_ptr'], res['txt_len'])
            try:
                text = bytes(txt_bytes).decode('utf-8')
                final.append({
                    "id": res['id'], 
                    "score": res['score'], 
                    "text": text
                })
            except:
                final.append({
                    "id": res['id'], 
                    "score": res['score'], 
                    "text": "[DECODE ERROR]"
                })
                
        return final

    def delete_entry(self, slot_id):
        """
        Deletes an entry by overwriting it with the LAST entry (Swap-and-Pop).
        This keeps the Red/Blue channels contiguous.
        Note: The Green (Text) channel remains fragmented until defrag() is called.
        """
        if not (0 <= slot_id < self.count): return
        
        last_id = self.count - 1
        
        if slot_id == last_id:
            self.count -= 1
            self._save()
            return

        # --- SWAP OPERATION ---
        
        meta_last = self._read_bytes(2, last_id * self.META_SIZE, self.META_SIZE)
        vec_ptr_last, txt_ptr_last, txt_len_last = struct.unpack('>III', bytes(meta_last))
        
        vec_last = self._read_bytes(0, vec_ptr_last, self.dim)
        self._write_bytes(0, slot_id * self.dim, vec_last)
        new_meta = struct.pack('>III', slot_id * self.dim, txt_ptr_last, txt_len_last)
        self._write_bytes(2, slot_id * self.META_SIZE, new_meta)
        
        self.count -= 1
        #self.defrag()
        self._save()

    def defrag(self):
        """
        Rebuilds the image to remove orphaned text from the Green Channel.
        """
        print(f"[{self.name}] Defragging (Rebuilding Green Channel)...")
        
        # 1. Read all active data into RAM
        active_entries = []
        for i in range(self.count):
            meta = self._read_bytes(2, i * self.META_SIZE, self.META_SIZE)
            vec_ptr, txt_ptr, txt_len = struct.unpack('>III', bytes(meta))
            
            vec = self._read_bytes(0, vec_ptr, self.dim)
            txt = self._read_bytes(1, txt_ptr, txt_len)
            
            active_entries.append((vec, txt))
            
        # 2. Reset Grid
        self._init_grid()
        
        # 3. Write back sequentially
        for vec_raw, txt_raw in active_entries:
            # We have raw bytes. We need to convert for write_entry
            # Vector: Raw bytes -> Float list
            vec_float = [(float(b) / 255.0 * 2.0) - 1.0 for b in vec_raw]
            
            # Text: Raw bytes -> String
            txt_str = bytes(txt_raw).decode('utf-8', errors='ignore')
            
            self.write_entry(txt_str, vec_float)
            
        print(f"[{self.name}] Defrag Complete. New Heap: {self.cursor_green}")

    def get_recent_entries(self, n=5):
        """Fetches the last N entries added."""
        entries = []
        curr = self.count - 1
        count = 0
        while curr >= 0 and count < n:
            meta = self._read_bytes(2, curr * self.META_SIZE, self.META_SIZE)
            _, txt_ptr, txt_len = struct.unpack('>III', bytes(meta))
            
            txt_bytes = self._read_bytes(1, txt_ptr, txt_len)
            try:
                entries.append(bytes(txt_bytes).decode('utf-8'))
            except:
                pass
                
            count += 1
            curr -= 1
        return list(reversed(entries))

    def dump_heap_content(self):
        """Returns all text content for debugging."""
        content = []
        for i in range(self.count):
            meta = self._read_bytes(2, i * self.META_SIZE, self.META_SIZE)
            _, txt_ptr, txt_len = struct.unpack('>III', bytes(meta))
            
            txt_bytes = self._read_bytes(1, txt_ptr, txt_len)
            text = bytes(txt_bytes).decode('utf-8', 'ignore')
            content.append(f"[{i}] {text}")
        return content

    def wipe(self):
        self._init_grid()
        self._save()

    def to_image_bytes(self):
        img = Image.fromarray(self.grid, 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf.getvalue()
