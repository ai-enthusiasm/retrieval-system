document.addEventListener("DOMContentLoaded", function () {
    // Khai báo và khởi tạo các biến
    const elements = {
        searchForm: document.getElementById('search-form'),
        queryInput: document.getElementById('text-query'),
    };

    let allFrames = [];
    let allMetadata = [];
    let searchType = 'image';
    let searchStartTime;

    // Xử lý sự kiện tìm kiếm
    elements.searchForm.addEventListener('submit', handleSearch);
    elements.queryInput.addEventListener('keypress', handleEnterKey);

    function handleSearch(event) {
        event.preventDefault();
        if (elements.queryInput.value.trim() !== '') {
            performSearch();
        } else {
            displayError('Please enter a query.');
        }
    }

    function handleEnterKey(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            performSearch();
        }
    }

    function performSearch() {
        searchStartTime = performance.now();
        console.log('🕒 Search started at:', new Date().toISOString());
        
        const formData = new FormData(elements.searchForm);
        const queryValue = elements.queryInput.value.trim();

        if (!queryValue) {
            displayError('Please enter a query.');
            return;
        }

        formData.append('query', queryValue);
        console.log('Sending search request with query:', queryValue);

        const fetchStartTime = performance.now();
        fetch('/search', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            const fetchEndTime = performance.now();
            console.log(`🕒 Network request took: ${(fetchEndTime - fetchStartTime).toFixed(2)}ms`);
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return response.json();
            } else {
                console.error('Unexpected content type:', contentType);
                return response.text().then(text => {
                    return text;
                });
            }
        })
        .then(data => {
            const dataReceiveTime = performance.now();
            console.log(`🕒 Time until data received: ${(dataReceiveTime - searchStartTime).toFixed(2)}ms`);
            
            if (typeof data === 'string') {
                document.getElementById('search-results').innerHTML = data;
            } else {
                displaySearchResults(data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            displayError('An error occurred while processing your request. Please try again.');
        });
    }

    //Hiển thị kết quả tìm kiếm
    function displaySearchResults(data) {
        const displayStartTime = performance.now();
        const searchResultsContainer = document.getElementById('search-results');
        if (!searchResultsContainer) {
            console.error('searchResultsContainer element not found.');
            return;
        }
        searchResultsContainer.innerHTML = '';
    
        if (data.error) {
            console.error('Error in data:', data.error);
            displayError(data.error);
            return;
        }
    
        allFrames = data.frame_paths || [];
        allMetadata = data.metadata_list || [];
        
        if (allFrames.length === 0) {
            displayError('No images found for your query.');
            return;
        }
    
        displayImageResults(allFrames, allMetadata);
        const displayEndTime = performance.now();
        console.log(`🕒 Time to process display data: ${(displayEndTime - displayStartTime).toFixed(2)}ms`);
    }

    //hiển thị kết quả tìm kiếm ảnh
    function displayImageResults(frames, metadata) {
        const imageStartTime = performance.now();
        const searchResultsContainer = document.getElementById('search-results');
        
        searchResultsContainer.innerHTML = '';
        let loadedImages = 0;
        
        if (frames.length > 0) {
            frames.forEach((framePath, index) => {
                
                const container = document.createElement('div');
                container.className = 'image-container';
        
                const img = document.createElement('img');
                img.src = framePath;
                img.alt = 'Frame Image';
                img.className = 'clickable-frame';
                
                // Add load event listener
                img.addEventListener('load', () => {
                    loadedImages++;
                    
                    if (loadedImages === frames.length) {
                        const totalTime = performance.now() - searchStartTime;
                        console.log(`🏁 All images loaded! Total search and display time: ${totalTime.toFixed(2)}ms`);
                        console.log(`📊 Performance Summary:
                            Total Time: ${totalTime.toFixed(2)}ms
                            Images Loaded: ${loadedImages}/${frames.length}
                        `);
                    }
                });
                
                img.addEventListener('click', () => {
                    showFrameModal(framePath, frames);
                });
                
                img.onerror = function() {
                    loadedImages++;
                    console.error(`Failed to load image ${index + 1}:`, framePath);
                    this.src = '/static/placeholder.jpg';
                    this.onerror = null;  // Prevent infinite loop
                };
                
                container.appendChild(img);
        
                const infoButton = document.createElement('button');
                infoButton.className = 'info-button';
                infoButton.textContent = 'Info';
                infoButton.addEventListener('click', function () {
                    showInfo(metadata[index]);
                });
                container.appendChild(infoButton);
        
                searchResultsContainer.appendChild(container);
            });
            
            const domUpdateTime = performance.now();
            console.log(`🕒 DOM updates completed in: ${(domUpdateTime - imageStartTime).toFixed(2)}ms`);
            
        } else {
            console.log("No frames to display");
            searchResultsContainer.innerHTML = '<p class="error-message">No images found. Try a different query.</p>';
        }
    }
    
    //hiển thị modal cho frame
    function showFrameModal(currentFrame, allFrames) {
        const modal = document.createElement('div');
        modal.className = 'frame-modal';
        modal.style.display = 'block';
        modal.innerHTML = `
            <span class="close-modal">&times;</span>
            <img src="${currentFrame}" alt="Frame" class="modal-content">
            <a class="prev">&#10094;</a>
            <a class="next">&#10095;</a>
        `;
        document.body.appendChild(modal);
    
        let currentIndex = allFrames.indexOf(currentFrame);
    
        function updateFrame(index) {
            if (index >= 0 && index < allFrames.length) {
                currentIndex = index;
                modal.querySelector('.modal-content').src = allFrames[currentIndex];
            }
        }
    
        modal.querySelector('.prev').addEventListener('click', () => updateFrame(currentIndex - 1));
        modal.querySelector('.next').addEventListener('click', () => updateFrame(currentIndex + 1));
    
        function handleKeyDown(e) {
            if (e.key === 'ArrowLeft') updateFrame(currentIndex - 1);
            if (e.key === 'ArrowRight') updateFrame(currentIndex + 1);
            if (e.key === 'Escape') closeModal();
        }
    
        document.addEventListener('keydown', handleKeyDown);
    
        function closeModal() {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.removeChild(modal);
        }
    
        modal.querySelector('.close-modal').addEventListener('click', closeModal);
    
        modal.addEventListener('click', (event) => {
            if (event.target === modal) {
                closeModal();
            }
        });
    }
    
    //hiển thị thông tin chi tiết của ảnh
    function showInfo(metadata) {
        let infoDetails = document.getElementById('info-details');
        const infoContent = `
            <h2>Image Information</h2>
            <p><strong>Folder Videos:</strong> ${metadata.video_folder}</p>
            <p><strong>Frame Number:</strong> ${metadata.frame_number}</p>
            <p><strong>Frame IDX:</strong> ${metadata.frame_idx}</p>
            <p><strong>PTS Time:</strong> ${metadata.pts_time}</p>
            <h3>Frame:</h3>
            <div class="frames">
                <img src="${metadata.frame_path}" alt="Frame Image" class="clickable-frame">
            </div>
        `;
        
        infoDetails.innerHTML = infoContent;
        
        document.getElementById('info-modal').style.display = 'block';
        
        const frame = infoDetails.querySelector('.clickable-frame');
        frame.addEventListener('click', function() {
            showFrameModal(this.src, [this.src]);
        });
    }

    //hiển thị thông báo lỗi
    function displayError(message) {
        const searchResultsContainer = document.getElementById('search-results');
        if (searchResultsContainer) {
            searchResultsContainer.innerHTML = `<p class="error-message">${message}</p>`;
        }
    }

    window.closeInfoModal = function () {
        const infoModal = document.getElementById('info-modal');
        if (infoModal) infoModal.style.display = 'none';
    };

    const closeButton = document.querySelector('.close-button');
    if (closeButton) {
        closeButton.addEventListener('click', closeInfoModal);
    }
});