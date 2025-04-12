document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewSection = document.getElementById('previewSection');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultSection = document.getElementById('resultSection');
    const loader = document.getElementById('loader');
    const resultData = document.getElementById('resultData');
    const deviceName = document.getElementById('deviceName');

    uploadArea.addEventListener('click', () => fileInput.click());
    
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    
    fileInput.addEventListener('change', handleFileSelect);
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });
    
    analyzeBtn.addEventListener('click', analyzeImage);
    
    function handleFileSelect() {
        const file = fileInput.files[0];
        
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                previewSection.style.display = 'block';
                resultSection.style.display = 'none';
            };
            
            reader.readAsDataURL(file);
        }
    }
    
    function analyzeImage() {
        resultSection.style.display = 'block';
        loader.style.display = 'block';
        resultData.style.display = 'none';
        
        const file = fileInput.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        fetch('/calculate-parameters', {
                method: 'POST',
                body: formData
            }
        )
            .then(value => value.json())
            .then(
            value => {
                const result = value
                console.log(result)
                deviceName.textContent = result.result.device_type;
                loader.style.display = 'none';
                resultData.style.display = 'block';
            }
        )
    }
});