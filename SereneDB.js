// SereneDB | In-browser vector database with multi-layer caching
// Supports IndexedDB for persistence, localStorage for caching, and in-memory operations

class SereneDB {
    #dbName;
    #db;
    #memoryCache;
    #cacheKey;
    #isDirty;
    #searchConfig;
    #metadata;

    /**
     * Create a new SereneDB instance
     * @param {string} dbName - Name of the database
     */
    constructor(dbName = 'SereneDB') {
        this.#dbName = dbName;
        this.#db = null;
        this.#memoryCache = new Map();
        this.#cacheKey = `sereneDB_cache_${dbName}`;
        this.#isDirty = false;
        this.#metadata = {
            vectorDimension: null,
            itemCount: 0,
            lastSync: null
        };
        this.#searchConfig = {
            metric: 'cosine', // 'cosine', 'euclidean', 'dot'
            threshold: 0.0,   // Minimum similarity threshold
            limit: 10         // Default result limit
        };
    }

    // ==================== INITIALIZATION ====================

    /**
     * Initialize SereneDB by creating IndexedDB and setting up event handlers
     * @returns {Promise<void>}
     */
    async create() {
        if (typeof window === 'undefined' || !window.indexedDB) {
            throw this.sereneDBErrorHandler('IndexedDB is not supported by this browser.');
        }

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.#dbName, 1);

            request.onerror = (event) => {
                const error = this.sereneDBErrorHandler(`Database error: ${event.target.errorCode}`);
                reject(error);
            };

            request.onupgradeneeded = (event) => {
                this.#db = event.target.result;
                
                // Create object store for vector data
                if (!this.#db.objectStoreNames.contains('vectors')) {
                    const objectStore = this.#db.createObjectStore('vectors', { keyPath: 'id' });
                    objectStore.createIndex('id', 'id', { unique: true });
                    objectStore.createIndex('timestamp', 'timestamp', { unique: false });
                }

                // Create object store for metadata
                if (!this.#db.objectStoreNames.contains('metadata')) {
                    this.#db.createObjectStore('metadata', { keyPath: 'key' });
                }
            };

            request.onsuccess = async (event) => {
                this.#db = event.target.result;
                
                // Setup abrupt closure handler
                this.#setupAbruptClosureHandler();
                
                // Try to load cache from localStorage
                await this.loadCacheFromSereneDB();
                
                resolve();
            };

            request.onblocked = () => {
                console.warn('SereneDB: Database upgrade blocked. Close other tabs with this database open.');
            };
        });
    }

    /**
     * Setup handler for abrupt page closure
     * @private
     */
    #setupAbruptClosureHandler() {
        if (typeof window === 'undefined') return;

        window.addEventListener('beforeunload', () => {
            this.handleSereneDBAbruptClosure();
        });

        window.addEventListener('pagehide', () => {
            this.handleSereneDBAbruptClosure();
        });
    }

    // ==================== DATA STORAGE & RETRIEVAL ====================

    /**
     * Load all vector data from IndexedDB into memory
     * @returns {Promise<void>}
     */
    async loadData() {
        if (!this.#db) {
            throw this.sereneDBErrorHandler('Database has not been created yet. Call create() first.');
        }

        return new Promise((resolve, reject) => {
            const transaction = this.#db.transaction(['vectors'], 'readonly');
            const objectStore = transaction.objectStore('vectors');
            const request = objectStore.getAll();

            request.onerror = (event) => {
                const error = this.sereneDBErrorHandler(`Failed to load data: ${event.target.error}`);
                reject(error);
            };

            request.onsuccess = (event) => {
                const results = event.target.result || [];
                this.#memoryCache.clear();
                
                results.forEach(item => {
                    this.#memoryCache.set(item.id, item);
                });

                this.#metadata.itemCount = results.length;
                
                // Detect vector dimension from first item
                if (results.length > 0 && results[0].vector) {
                    this.#metadata.vectorDimension = results[0].vector.length;
                }

                this.#metadata.lastSync = Date.now();
                resolve();
            };
        });
    }

    /**
     * Save vector data to IndexedDB
     * @param {Object|Array} vectorData - Single vector object or array of vectors
     * @param {string} vectorData.id - Unique identifier for the vector
     * @param {Array<number>} vectorData.vector - The vector data
     * @param {Object} vectorData.metadata - Optional metadata
     * @returns {Promise<void>}
     */
    async saveDataToSereneDB(vectorData) {
        if (!this.#db) {
            throw this.sereneDBErrorHandler('Database has not been created yet. Call create() first.');
        }

        const items = Array.isArray(vectorData) ? vectorData : [vectorData];
        
        // Validate input
        for (const item of items) {
            if (!item.id) {
                throw this.sereneDBErrorHandler('Each vector item must have an id property.');
            }
            if (!item.vector || !Array.isArray(item.vector)) {
                throw this.sereneDBErrorHandler(`Vector item ${item.id} must have a vector array.`);
            }
            if (this.#metadata.vectorDimension === null) {
                this.#metadata.vectorDimension = item.vector.length;
            } else if (item.vector.length !== this.#metadata.vectorDimension) {
                throw this.sereneDBErrorHandler(
                    `Vector dimension mismatch. Expected ${this.#metadata.vectorDimension}, got ${item.vector.length}.`
                );
            }
        }

        return new Promise((resolve, reject) => {
            const transaction = this.#db.transaction(['vectors'], 'readwrite');
            const objectStore = transaction.objectStore('vectors');

            transaction.onerror = (event) => {
                const error = this.sereneDBErrorHandler(`Transaction error: ${event.target.error}`);
                reject(error);
            };

            let completed = 0;
            const total = items.length;

            items.forEach(item => {
                const record = {
                    id: item.id,
                    vector: item.vector,
                    metadata: item.metadata || {},
                    timestamp: item.timestamp || Date.now()
                };

                const request = objectStore.put(record);

                request.onsuccess = () => {
                    // Update memory cache
                    this.#memoryCache.set(record.id, record);
                    completed++;
                    
                    if (completed === total) {
                        this.#isDirty = true;
                        this.#metadata.itemCount = this.#memoryCache.size;
                        this.updateCacheInSereneDB();
                        resolve();
                    }
                };

                request.onerror = (event) => {
                    const error = this.sereneDBErrorHandler(`Failed to save item ${item.id}: ${event.target.error}`);
                    reject(error);
                };
            });

            // Handle empty array case
            if (total === 0) {
                resolve();
            }
        });
    }

    /**
     * Get a single vector by ID
     * @param {string} id - Vector ID
     * @returns {Object|null} Vector data or null if not found
     */
    get(id) {
        return this.#memoryCache.get(id) || null;
    }

    /**
     * Delete a vector by ID
     * @param {string} id - Vector ID to delete
     * @returns {Promise<boolean>} True if deleted, false if not found
     */
    async delete(id) {
        if (!this.#db) {
            throw this.sereneDBErrorHandler('Database has not been created yet. Call create() first.');
        }

        if (!this.#memoryCache.has(id)) {
            return false;
        }

        return new Promise((resolve, reject) => {
            const transaction = this.#db.transaction(['vectors'], 'readwrite');
            const objectStore = transaction.objectStore('vectors');
            const request = objectStore.delete(id);

            request.onsuccess = () => {
                this.#memoryCache.delete(id);
                this.#isDirty = true;
                this.#metadata.itemCount = this.#memoryCache.size;
                this.updateCacheInSereneDB();
                resolve(true);
            };

            request.onerror = (event) => {
                const error = this.sereneDBErrorHandler(`Failed to delete ${id}: ${event.target.error}`);
                reject(error);
            };
        });
    }

    /**
     * Delete multiple vectors by IDs
     * @param {Array<string>} ids - Array of vector IDs to delete
     * @returns {Promise<number>} Number of deleted items
     */
    async deleteMany(ids) {
        let deleted = 0;
        for (const id of ids) {
            if (await this.delete(id)) {
                deleted++;
            }
        }
        return deleted;
    }

    /**
     * Clear all vectors from the database
     * @returns {Promise<void>}
     */
    async clear() {
        if (!this.#db) {
            throw this.sereneDBErrorHandler('Database has not been created yet. Call create() first.');
        }

        return new Promise((resolve, reject) => {
            const transaction = this.#db.transaction(['vectors'], 'readwrite');
            const objectStore = transaction.objectStore('vectors');
            const request = objectStore.clear();

            request.onsuccess = () => {
                this.#memoryCache.clear();
                this.#isDirty = true;
                this.#metadata.itemCount = 0;
                this.#metadata.vectorDimension = null;
                this.clearSereneDBCache();
                resolve();
            };

            request.onerror = (event) => {
                const error = this.sereneDBErrorHandler(`Failed to clear database: ${event.target.error}`);
                reject(error);
            };
        });
    }

    // ==================== CACHE MANAGEMENT ====================

    /**
     * Load cached data from localStorage into memory
     * @returns {Promise<void>}
     */
    async loadCacheFromSereneDB() {
        if (typeof localStorage === 'undefined') {
            console.warn('SereneDB: localStorage not available. Skipping cache load.');
            return;
        }

        try {
            const cached = localStorage.getItem(this.#cacheKey);
            if (cached) {
                const parsed = JSON.parse(cached);
                
                if (parsed.data && Array.isArray(parsed.data)) {
                    this.#memoryCache.clear();
                    parsed.data.forEach(item => {
                        this.#memoryCache.set(item.id, item);
                    });
                    this.#metadata.itemCount = parsed.data.length;
                }

                if (parsed.metadata) {
                    this.#metadata = { ...this.#metadata, ...parsed.metadata };
                }

                if (parsed.searchConfig) {
                    this.#searchConfig = { ...this.#searchConfig, ...parsed.searchConfig };
                }
            }
        } catch (error) {
            console.warn('SereneDB: Failed to load cache from localStorage:', error.message);
        }
    }

    /**
     * Update localStorage with current in-memory data
     */
    updateCacheInSereneDB() {
        if (typeof localStorage === 'undefined') {
            console.warn('SereneDB: localStorage not available. Skipping cache update.');
            return;
        }

        try {
            const cacheData = {
                data: Array.from(this.#memoryCache.values()),
                metadata: {
                    ...this.#metadata,
                    lastCacheUpdate: Date.now()
                },
                searchConfig: this.#searchConfig
            };

            localStorage.setItem(this.#cacheKey, JSON.stringify(cacheData));
            this.#isDirty = false;
        } catch (error) {
            // Handle quota exceeded
            if (error.name === 'QuotaExceededError') {
                console.warn('SereneDB: localStorage quota exceeded. Attempting to compress cache.');
                this.#compressAndCache();
            } else {
                console.warn('SereneDB: Failed to update cache:', error.message);
            }
        }
    }

    /**
     * Compress and cache data when localStorage quota is exceeded
     * @private
     */
    #compressAndCache() {
        try {
            // Store only essential data (IDs and timestamps) to reduce size
            const compressedData = {
                data: Array.from(this.#memoryCache.values()).map(item => ({
                    id: item.id,
                    timestamp: item.timestamp
                })),
                metadata: {
                    ...this.#metadata,
                    compressed: true,
                    lastCacheUpdate: Date.now()
                }
            };

            localStorage.setItem(this.#cacheKey, JSON.stringify(compressedData));
            this.#isDirty = false;
        } catch (error) {
            console.warn('SereneDB: Failed to compress cache. Cache disabled for this session.');
        }
    }

    /**
     * Clear the localStorage cache
     */
    clearSereneDBCache() {
        if (typeof localStorage === 'undefined') {
            return;
        }

        try {
            localStorage.removeItem(this.#cacheKey);
        } catch (error) {
            console.warn('SereneDB: Failed to clear cache:', error.message);
        }
    }

    /**
     * Synchronize all storage layers
     * @returns {Promise<void>}
     */
    async syncSereneDBCacheAndDatabase() {
        if (!this.#db) {
            throw this.sereneDBErrorHandler('Database has not been created yet. Call create() first.');
        }

        // First, load from IndexedDB to get the most recent data
        await this.loadData();
        
        // Then update localStorage cache
        this.updateCacheInSereneDB();
        
        this.#metadata.lastSync = Date.now();
    }

    // ==================== VECTOR SEARCH ====================

    /**
     * Perform vector similarity search
     * @param {Array<number>} queryVector - The query vector
     * @param {Object} options - Search options
     * @param {number} options.limit - Maximum number of results (default: 10)
     * @param {number} options.threshold - Minimum similarity threshold (default: 0)
     * @param {string} options.metric - Distance metric: 'cosine', 'euclidean', 'dot' (default: 'cosine')
     * @returns {Array} Array of search results with similarity scores
     */
    performSereneDBVectorSearch(queryVector, options = {}) {
        const {
            limit = this.#searchConfig.limit,
            threshold = this.#searchConfig.threshold,
            metric = this.#searchConfig.metric
        } = options;

        if (!Array.isArray(queryVector)) {
            throw this.sereneDBErrorHandler('Query vector must be an array.');
        }

        if (this.#memoryCache.size === 0) {
            return [];
        }

        const results = [];

        for (const [id, item] of this.#memoryCache) {
            if (!item.vector || item.vector.length !== queryVector.length) {
                continue;
            }

            const similarity = this.#calculateSimilarity(queryVector, item.vector, metric);
            
            if (similarity >= threshold) {
                results.push({
                    id,
                    vector: item.vector,
                    metadata: item.metadata,
                    similarity,
                    timestamp: item.timestamp
                });
            }
        }

        // Sort by similarity (descending for cosine/dot, ascending for euclidean)
        if (metric === 'euclidean') {
            results.sort((a, b) => a.similarity - b.similarity);
        } else {
            results.sort((a, b) => b.similarity - a.similarity);
        }

        return results.slice(0, limit);
    }

    /**
     * Calculate similarity between two vectors
     * @param {Array<number>} v1 - First vector
     * @param {Array<number>} v2 - Second vector
     * @param {string} metric - Distance metric
     * @returns {number} Similarity score
     * @private
     */
    #calculateSimilarity(v1, v2, metric) {
        switch (metric) {
            case 'cosine':
                return this.#cosineSimilarity(v1, v2);
            case 'euclidean':
                return this.#euclideanDistance(v1, v2);
            case 'dot':
                return this.#dotProduct(v1, v2);
            default:
                return this.#cosineSimilarity(v1, v2);
        }
    }

    /**
     * Calculate cosine similarity between two vectors
     * @param {Array<number>} v1 - First vector
     * @param {Array<number>} v2 - Second vector
     * @returns {number} Cosine similarity (-1 to 1)
     * @private
     */
    #cosineSimilarity(v1, v2) {
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;

        for (let i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
        return denominator === 0 ? 0 : dotProduct / denominator;
    }

    /**
     * Calculate Euclidean distance between two vectors
     * @param {Array<number>} v1 - First vector
     * @param {Array<number>} v2 - Second vector
     * @returns {number} Euclidean distance (lower is more similar)
     * @private
     */
    #euclideanDistance(v1, v2) {
        let sum = 0;
        for (let i = 0; i < v1.length; i++) {
            const diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Calculate dot product between two vectors
     * @param {Array<number>} v1 - First vector
     * @param {Array<number>} v2 - Second vector
     * @returns {number} Dot product
     * @private
     */
    #dotProduct(v1, v2) {
        let product = 0;
        for (let i = 0; i < v1.length; i++) {
            product += v1[i] * v2[i];
        }
        return product;
    }

    /**
     * Optimize search performance based on data characteristics
     * @param {Object} options - Optimization options
     */
    optimizeSereneDBSearchPerformance(options = {}) {
        const {
            useQuantization = false,
            quantizationBits = 8,
            useApproximateSearch = false,
            approximateThreshold = 1000
        } = options;

        // Enable quantization for memory efficiency
        if (useQuantization && this.#memoryCache.size > 0) {
            this.#applyQuantization(quantizationBits);
        }

        // Update search config for approximate search on large datasets
        if (useApproximateSearch && this.#memoryCache.size > approximateThreshold) {
            this.#searchConfig.approximateSearch = true;
            this.#searchConfig.approximateThreshold = approximateThreshold;
        }

        // Pre-compute norms for cosine similarity optimization
        this.#precomputeNorms();
    }

    /**
     * Apply quantization to vectors for memory efficiency
     * @param {number} bits - Number of bits for quantization
     * @private
     */
    #applyQuantization(bits) {
        const maxVal = Math.pow(2, bits - 1) - 1;
        const minVal = -Math.pow(2, bits - 1);

        for (const [id, item] of this.#memoryCache) {
            if (item.vector) {
                // Find min/max for normalization
                const vecMin = Math.min(...item.vector);
                const vecMax = Math.max(...item.vector);
                const range = vecMax - vecMin || 1;

                item.quantizedVector = item.vector.map(v => {
                    const normalized = (v - vecMin) / range;
                    return Math.round(normalized * (maxVal - minVal) + minVal);
                });
                item.quantizationRange = { min: vecMin, max: vecMax };
            }
        }
    }

    /**
     * Pre-compute norms for faster cosine similarity
     * @private
     */
    #precomputeNorms() {
        for (const [id, item] of this.#memoryCache) {
            if (item.vector && !item.norm) {
                item.norm = Math.sqrt(
                    item.vector.reduce((sum, v) => sum + v * v, 0)
                );
            }
        }
    }

    // ==================== ERROR HANDLING ====================

    /**
     * Centralized error handler
     * @param {string|Error} error - Error message or object
     * @returns {Error} Formatted error object
     */
    sereneDBErrorHandler(error) {
        const timestamp = new Date().toISOString();
        const message = typeof error === 'string' ? error : error.message;
        
        const formattedError = new Error(`[SereneDB ${timestamp}] ${message}`);
        formattedError.name = 'SereneDBError';
        formattedError.timestamp = timestamp;
        formattedError.originalError = error;

        console.error(formattedError.message);
        
        return formattedError;
    }

    // ==================== ABRUPT CLOSURE HANDLING ====================

    /**
     * Handle abrupt closure by saving data to localStorage
     */
    handleSereneDBAbruptClosure() {
        if (!this.#isDirty && this.#memoryCache.size === 0) {
            return;
        }

        // Synchronous save to localStorage for abrupt closure
        if (typeof localStorage !== 'undefined') {
            try {
                const emergencyCache = {
                    data: Array.from(this.#memoryCache.values()),
                    metadata: {
                        ...this.#metadata,
                        emergencySave: true,
                        emergencySaveTime: Date.now()
                    }
                };
                localStorage.setItem(`${this.#cacheKey}_emergency`, JSON.stringify(emergencyCache));
            } catch (error) {
                console.error('SereneDB: Failed emergency save:', error.message);
            }
        }
    }

    /**
     * Recover from emergency save after abrupt closure
     * @returns {Promise<boolean>} True if recovery was successful
     */
    async recoverFromEmergencySave() {
        if (typeof localStorage === 'undefined') {
            return false;
        }

        try {
            const emergencyCache = localStorage.getItem(`${this.#cacheKey}_emergency`);
            if (emergencyCache) {
                const parsed = JSON.parse(emergencyCache);
                
                if (parsed.data && Array.isArray(parsed.data)) {
                    await this.saveDataToSereneDB(parsed.data);
                    localStorage.removeItem(`${this.#cacheKey}_emergency`);
                    return true;
                }
            }
        } catch (error) {
            console.warn('SereneDB: Failed to recover from emergency save:', error.message);
        }

        return false;
    }

    // ==================== MEMORY MANAGEMENT ====================

    /**
     * Unload data from memory to free resources
     * @param {boolean} saveToCache - Whether to save to localStorage before unloading
     */
    unloadSereneDBDataFromMemory(saveToCache = true) {
        if (saveToCache) {
            this.updateCacheInSereneDB();
        }

        this.#memoryCache.clear();
        this.#metadata.itemCount = 0;
    }

    /**
     * Get memory usage statistics
     * @returns {Object} Memory usage info
     */
    getMemoryStats() {
        let estimatedSize = 0;
        
        for (const [id, item] of this.#memoryCache) {
            // Rough estimate of memory usage
            estimatedSize += id.length * 2; // String characters (UTF-16)
            if (item.vector) {
                estimatedSize += item.vector.length * 8; // Float64 numbers
            }
            if (item.metadata) {
                estimatedSize += JSON.stringify(item.metadata).length * 2;
            }
        }

        return {
            itemCount: this.#memoryCache.size,
            estimatedBytes: estimatedSize,
            estimatedKB: Math.round(estimatedSize / 1024 * 100) / 100,
            estimatedMB: Math.round(estimatedSize / 1024 / 1024 * 100) / 100,
            vectorDimension: this.#metadata.vectorDimension
        };
    }

    // ==================== UTILITY METHODS ====================

    /**
     * Get the number of vectors in the database
     * @returns {number} Count of vectors
     */
    count() {
        return this.#memoryCache.size;
    }

    /**
     * Check if a vector exists
     * @param {string} id - Vector ID
     * @returns {boolean} True if exists
     */
    has(id) {
        return this.#memoryCache.has(id);
    }

    /**
     * Get all vector IDs
     * @returns {Array<string>} Array of IDs
     */
    getAllIds() {
        return Array.from(this.#memoryCache.keys());
    }

    /**
     * Get all vectors
     * @returns {Array<Object>} Array of all vector objects
     */
    getAll() {
        return Array.from(this.#memoryCache.values());
    }

    /**
     * Set search configuration
     * @param {Object} config - Search configuration
     */
    setSearchConfig(config) {
        this.#searchConfig = { ...this.#searchConfig, ...config };
    }

    /**
     * Get current search configuration
     * @returns {Object} Search configuration
     */
    getSearchConfig() {
        return { ...this.#searchConfig };
    }

    /**
     * Get database metadata
     * @returns {Object} Metadata object
     */
    getMetadata() {
        return { ...this.#metadata };
    }

    /**
     * Export database to JSON
     * @returns {Object} JSON representation of the database
     */
    export() {
        return {
            dbName: this.#dbName,
            data: Array.from(this.#memoryCache.values()),
            metadata: this.#metadata,
            searchConfig: this.#searchConfig,
            exportTime: Date.now()
        };
    }

    /**
     * Import data from JSON export
     * @param {Object} data - JSON data from export
     * @returns {Promise<void>}
     */
    async import(data) {
        if (!data || !data.data || !Array.isArray(data.data)) {
            throw this.sereneDBErrorHandler('Invalid import data format.');
        }

        await this.saveDataToSereneDB(data.data);

        if (data.metadata) {
            this.#metadata = { ...this.#metadata, ...data.metadata };
        }

        if (data.searchConfig) {
            this.#searchConfig = { ...this.#searchConfig, ...data.searchConfig };
        }
    }

    /**
     * Batch insert vectors with progress callback
     * @param {Array<Object>} vectors - Array of vectors to insert
     * @param {Function} onProgress - Progress callback (current, total)
     * @param {number} batchSize - Number of vectors per batch
     * @returns {Promise<void>}
     */
    async batchInsert(vectors, onProgress = null, batchSize = 100) {
        const total = vectors.length;
        
        for (let i = 0; i < total; i += batchSize) {
            const batch = vectors.slice(i, Math.min(i + batchSize, total));
            await this.saveDataToSereneDB(batch);
            
            if (onProgress) {
                onProgress(Math.min(i + batchSize, total), total);
            }
        }
    }

    /**
     * Close the database connection
     */
    close() {
        if (this.#db) {
            this.#db.close();
            this.#db = null;
        }
        
        // Save to cache before closing
        this.updateCacheInSereneDB();
    }

    /**
     * Destroy the database completely
     * @returns {Promise<void>}
     */
    async destroy() {
        this.close();
        this.unloadSereneDBDataFromMemory(false);
        this.clearSereneDBCache();

        return new Promise((resolve, reject) => {
            const request = indexedDB.deleteDatabase(this.#dbName);

            request.onsuccess = () => {
                resolve();
            };

            request.onerror = (event) => {
                const error = this.sereneDBErrorHandler(`Failed to destroy database: ${event.target.error}`);
                reject(error);
            };
        });
    }
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SereneDB;
}

if (typeof window !== 'undefined') {
    window.SereneDB = SereneDB;
}
