// SereneDB | in-browser vector database + in-browser ONNX runtime.
class SereneDB {
    constructor(dbName) {
        this.dbName = dbName;
        this.db = null;
        this.memoryCache = {};
    }

    async create() {
        if (!window.indexedDB) {
            console.error("IndexedDB is not supported by this browser.");
            return;
        }

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName);

            request.onerror = (event) => {
                console.error("Database error: " + event.target.errorCode);
                reject(event.target.errorCode);
            };

            request.onupgradeneeded = (event) => {
                this.db = event.target.result;
                // Create an object store for vector data  
                const objectStore = this.db.createObjectStore("vectors", { keyPath: "id" });
                // Define any necessary indexes  
                objectStore.createIndex("vector", "vector", { unique: false });
            };

            request.onsuccess = (event) => {
                this.db = event.target.result;
                resolve();
            };
        });
    }

    async loadData() {
        return new Promise((resolve, reject) => {
            if (!this.db) {
                console.error("Database has not been created yet.");
                reject("Database has not been created yet.");
                return;
            }

            const transaction = this.db.transaction(["vectors"], "readonly");

            if (!transaction) {
                console.error("Failed to create a transaction.");
                reject("Failed to create a transaction.");
                return;
            }

            const objectStore = transaction.objectStore("vectors");

            if (!objectStore) {
                console.error("Failed to access the object store.");
                reject("Failed to access the object store.");
                return;
            }

            const request = objectStore.getAll();

            request.onerror = (event) => {
                console.error("Database error: " + event.target.errorCode);
                reject(event.target.errorCode);
            };

            request.onsuccess = (event) => {
                if (!event.target.result) {
                    console.error("Failed to load data from the database.");
                    reject("Failed to load data from the database.");
                    return;
                }

                this.memoryCache = event.target.result;
                resolve();
            };
        });
    }
}