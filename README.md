# SereneDB

SereneDB is a fast, reliable, open-source in-browser vector database designed for efficient storage, retrieval, and similarity search of vector embeddings. It leverages IndexedDB for persistent storage, localStorage for caching, and in-memory operations for lightning-fast vector searches.

## ✨ Features

- **🚀 Fast Vector Search** - In-memory operations with cosine, euclidean, and dot product similarity metrics
- **💾 Multi-Layer Storage** - IndexedDB for persistence, localStorage for caching, memory for speed
- **🔄 Auto-Sync** - Automatic synchronization between storage layers
- **🛡️ Data Integrity** - Handles abrupt closures with emergency save to localStorage
- **📦 Batch Operations** - Efficient batch insert with progress callbacks
- **🔧 Configurable** - Customizable search parameters and optimization options
- **📊 Memory Management** - Load/unload data to manage memory efficiently
- **🌐 Browser Native** - No external dependencies, works in all modern browsers

## 🚀 Quick Start

```javascript
// Create a new SereneDB instance
const db = new SereneDB('myVectorDB');

// Initialize the database
await db.create();

// Insert vectors
await db.saveDataToSereneDB({
    id: 'doc1',
    vector: [0.1, 0.2, 0.3, 0.4],
    metadata: { title: 'Document 1', category: 'tech' }
});

// Batch insert
await db.saveDataToSereneDB([
    { id: 'doc2', vector: [0.2, 0.3, 0.4, 0.5], metadata: { title: 'Document 2' } },
    { id: 'doc3', vector: [0.3, 0.4, 0.5, 0.6], metadata: { title: 'Document 3' } }
]);

// Perform vector search
const results = db.performSereneDBVectorSearch([0.1, 0.2, 0.3, 0.4], {
    limit: 5,
    threshold: 0.8,
    metric: 'cosine'
});

console.log(results);
// [{ id: 'doc1', similarity: 1.0, metadata: {...}, ... }, ...]
```

## 📖 API Reference

### Initialization

#### `constructor(dbName = 'SereneDB')`
Create a new SereneDB instance.

```javascript
const db = new SereneDB('myDatabase');
```

#### `async create()`
Initialize the database. Must be called before any operations.

```javascript
await db.create();
```

### Data Operations

#### `async saveDataToSereneDB(vectorData)`
Store or update vector data.

```javascript
// Single vector
await db.saveDataToSereneDB({
    id: 'unique-id',
    vector: [0.1, 0.2, 0.3],
    metadata: { any: 'data' }  // optional
});

// Multiple vectors
await db.saveDataToSereneDB([
    { id: 'id1', vector: [0.1, 0.2, 0.3] },
    { id: 'id2', vector: [0.4, 0.5, 0.6] }
]);
```

#### `async loadData()`
Load all vector data from IndexedDB into memory.

```javascript
await db.loadData();
```

#### `get(id)`
Retrieve a single vector by ID.

```javascript
const vector = db.get('my-id');
```

#### `has(id)`
Check if a vector exists.

```javascript
if (db.has('my-id')) {
    // Vector exists
}
```

#### `async delete(id)`
Delete a vector by ID.

```javascript
await db.delete('my-id');
```

#### `async deleteMany(ids)`
Delete multiple vectors.

```javascript
await db.deleteMany(['id1', 'id2', 'id3']);
```

#### `async clear()`
Remove all vectors from the database.

```javascript
await db.clear();
```

### Search Operations

#### `performSereneDBVectorSearch(queryVector, options)`
Perform similarity search on in-memory data.

```javascript
const results = db.performSereneDBVectorSearch([0.1, 0.2, 0.3], {
    limit: 10,           // Max results (default: 10)
    threshold: 0.5,      // Min similarity (default: 0)
    metric: 'cosine'     // 'cosine', 'euclidean', or 'dot'
});
```

**Distance Metrics:**
- `'cosine'` - Cosine similarity (default). Range: -1 to 1. Higher is more similar.
- `'euclidean'` - Euclidean distance. Lower is more similar.
- `'dot'` - Dot product. Higher is more similar.

#### `setSearchConfig(config)`
Set default search parameters.

```javascript
db.setSearchConfig({
    limit: 20,
    metric: 'euclidean',
    threshold: 0.1
});
```

#### `optimizeSereneDBSearchPerformance(options)`
Apply performance optimizations.

```javascript
db.optimizeSereneDBSearchPerformance({
    useQuantization: true,       // Reduce memory usage
    quantizationBits: 8,         // Quantization precision
    useApproximateSearch: true,  // For large datasets
    approximateThreshold: 1000   // When to use approximate search
});
```

### Cache Management

#### `async loadCacheFromSereneDB()`
Load cached data from localStorage into memory.

```javascript
await db.loadCacheFromSereneDB();
```

#### `updateCacheInSereneDB()`
Save current in-memory data to localStorage cache.

```javascript
db.updateCacheInSereneDB();
```

#### `clearSereneDBCache()`
Clear the localStorage cache.

```javascript
db.clearSereneDBCache();
```

#### `async syncSereneDBCacheAndDatabase()`
Synchronize all storage layers (IndexedDB ↔ localStorage ↔ memory).

```javascript
await db.syncSereneDBCacheAndDatabase();
```

### Memory Management

#### `unloadSereneDBDataFromMemory(saveToCache = true)`
Clear memory and optionally save to cache.

```javascript
db.unloadSereneDBDataFromMemory(true);  // Save to cache before unloading
```

#### `getMemoryStats()`
Get memory usage statistics.

```javascript
const stats = db.getMemoryStats();
console.log(stats);
// { itemCount: 100, estimatedBytes: 12800, estimatedKB: 12.5, estimatedMB: 0.01 }
```

### Utility Methods

#### `count()`
Get the number of vectors in memory.

```javascript
const total = db.count();
```

#### `getAllIds()`
Get all vector IDs.

```javascript
const ids = db.getAllIds();
```

#### `getAll()`
Get all vectors.

```javascript
const vectors = db.getAll();
```

#### `getMetadata()`
Get database metadata.

```javascript
const meta = db.getMetadata();
// { vectorDimension: 128, itemCount: 100, lastSync: 1234567890 }
```

#### `export()`
Export database to JSON.

```javascript
const data = db.export();
// { dbName: 'myDB', data: [...], metadata: {...}, searchConfig: {...} }
```

#### `async import(data)`
Import data from JSON export.

```javascript
await db.import(exportedData);
```

#### `async batchInsert(vectors, onProgress, batchSize)`
Insert large amounts of data with progress tracking.

```javascript
await db.batchInsert(vectors, (current, total) => {
    console.log(`Progress: ${current}/${total}`);
}, 100);
```

### Lifecycle

#### `close()`
Close the database connection.

```javascript
db.close();
```

#### `async destroy()`
Permanently delete the database.

```javascript
await db.destroy();
```

### Error Handling

#### `sereneDBErrorHandler(error)`
Centralized error handler. All errors are logged and wrapped in a SereneDBError.

```javascript
try {
    await db.saveDataToSereneDB(invalidData);
} catch (error) {
    console.error(error.message);  // [SereneDB 2024-01-01T00:00:00.000Z] Error message
}
```

### Data Integrity

#### `handleSereneDBAbruptClosure()`
Automatically called on page unload to save data to emergency cache.

#### `async recoverFromEmergencySave()`
Recover data from emergency save after abrupt closure.

```javascript
const recovered = await db.recoverFromEmergencySave();
if (recovered) {
    console.log('Data recovered from emergency save!');
}
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SereneDB                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Memory     │◄──►│ localStorage │◄──►│  IndexedDB   │  │
│  │   (Fast)     │    │   (Cache)    │    │ (Persistent) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         ▲                   ▲                    ▲          │
│         │                   │                    │          │
│         └───────────────────┴────────────────────┘          │
│                      Auto Sync                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Performance Tips

1. **Use batch operations** for inserting large datasets
2. **Call `optimizeSereneDBSearchPerformance()`** after loading data
3. **Use appropriate distance metrics** for your use case
4. **Set thresholds** to filter low-similarity results early
5. **Unload memory** when not actively searching

## 🧪 Testing

Open `test.html` in a browser to run the test suite.

```bash
# If you have a local server
open test.html
# or
npx serve .
```

## 📝 Data Structure

Each vector entry has the following structure:

```javascript
{
    id: 'unique-string-id',     // Required: Unique identifier
    vector: [0.1, 0.2, ...],    // Required: Array of numbers
    metadata: {                 // Optional: Any additional data
        title: 'Document Title',
        category: 'tech',
        // ... any other fields
    },
    timestamp: 1234567890       // Auto-generated: Creation/update time
}
```

## 🤝 Contributing

SereneDB is open-source! Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

SereneDB is released under the [MIT License](LICENSE).

---

Made with ❤️ for the vector search community
