
document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('send-button');
    const uploadButton = document.getElementById('upload-button');
    const resumeUpload = document.getElementById("resume-upload");
    const fileNameDisplay = document.getElementById("file-name");

    if (resumeUpload) {
        resumeUpload.addEventListener("change", function () {
            const fileName = this.files[0] ? this.files[0].name : "선택된 파일 없음";
            fileNameDisplay.textContent = fileName;
        });
    }
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
    
    if (uploadButton) {
        uploadButton.addEventListener('click', uploadResume);
    }

    // Enter key for sending message
    const userInput = document.getElementById('user-input');
    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault(); // 기본 Enter 동작 방지
                sendMessage();
            }
        });
    }
    // 초기 자동 응답 메시지 추가
    addMessage('안녕하세요! 무엇을 도와드릴까요?', 'bot-message');
});

async function sendMessage() {
    const userInput = document.getElementById('user-input');

    if (!userInput.value.trim()) return;

    addMessage(userInput.value, 'user-message');

    try {
        const response = await fetch('/retrieve_and_answer/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: userInput.value })
        });
    
        const data = await response.json();
        console.log("Server Response:", data);
    
        if (data && data.answer) {
            // Preprocess data.answer before displaying it
            const prettyAnswer = preprocessAnswer(data.answer);
            addMessage(prettyAnswer, 'bot-message'); // Display prettified bot response in the chat window
            updateRecommendations(data.answer); // Update recommendations with the original data
        } else {
            addMessage('추천된 후보자가 없습니다.', 'bot-message');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('요청을 처리하는 중에 오류가 발생했습니다.', 'bot-message');
    }

    userInput.value = ''; // Clear input field
}

// Function to preprocess the answer into a more readable format
function preprocessAnswer(answer) {
    // Assume answer is a JSON string or an object
    let parsedAnswer;
    if (typeof answer === 'string') {
        try {
            parsedAnswer = JSON.parse(answer);
        } catch (e) {
            console.error('Failed to parse answer:', e);
            return answer; // Return the original answer if parsing fails
        }
    } else {
        parsedAnswer = answer;
    }

    // Create a formatted output based on parsedAnswer structure
    let prettyOutput = '';
    if (Array.isArray(parsedAnswer)) {
        parsedAnswer.forEach((item, index) => {
            prettyOutput += `\n후보자 ${index + 1}:\n`;
            Object.keys(item).forEach(key => {
                if (key !== '경력' && !key.endsWith('score')) { // Exclude '경력' field and fields ending with 'score'
                    prettyOutput += `${key}: ${item[key]}\n`;
                }
            });
        });
    } else {
        Object.keys(parsedAnswer).forEach(key => {
            if (key !== '경력' && !key.endsWith('score')) { // Exclude '경력' field and fields ending with 'score'
                prettyOutput += `${key}: ${parsedAnswer[key]}\n`;
            }
        });
    }

    return prettyOutput;
}

// Add chat message to window
function addMessage(message, className) {
    const chatWindow = document.getElementById('chat-window');
    const messageContainer = document.createElement('div');
    messageContainer.className = `message-container ${className}`;
     // 아이콘 추가
    // const icon = document.createElement('div');
    // icon.className = 'icon';
    // icon.innerHTML = className === 'user-message' ? '👤' : '🤖';
    // 아이콘 추가 (innerHTML로 이미지 삽입)
    const icon = document.createElement('div');
    icon.className = 'icon';
    // 사용자와 봇에 따라 다른 아이콘 추가
    if (className === 'user-message') {
        icon.innerHTML = '<img src="/static/images/user.png" alt="User" style="width: 24px; height: 24px;">';
    } else {
        icon.innerHTML = '<img src="/static/images/bot.png" alt="Bot" style="width: 24px; height: 24px;">';
    }
     // 메시지 내용 추가
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    messageDiv.innerHTML = message.replace(/\n/g, '<br>');
    
    // 아이콘과 메시지를 컨테이너에 추가
    messageContainer.appendChild(icon);
    messageContainer.appendChild(messageDiv);
    // 봇 메시지와 사용자 메시지에 따라 순서 지정
    if (className === 'bot-message') {
        messageContainer.appendChild(icon); // 아이콘 먼저
        messageContainer.appendChild(messageDiv); // 메시지 나중
    } else {
        messageContainer.appendChild(messageDiv); // 메시지 먼저
        messageContainer.appendChild(icon); // 아이콘 나중
    }

    // 메시지 컨테이너를 채팅 창에 추가
    chatWindow.appendChild(messageContainer);
    chatWindow.scrollTop = chatWindow.scrollHeight; 
}

async function uploadResume() {
    const fileInput = document.getElementById('resume-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('업로드할 파일을 선택해주세요.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            alert('파일이 성공적으로 업로드되었습니다.');
        } else {
            alert(`파일 업로드 오류: ${data.error}`);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('파일 업로드 중 오류가 발생했습니다.');
    }
}

function updateRecommendations(recommendations) {
    const jsonContent = document.getElementById('json-content');
    jsonContent.innerHTML = ''; // 기존 내용을 지움

    const tabButtonsContainer = document.querySelector('.inner-tabs');
    tabButtonsContainer.innerHTML = ''; // 기존 탭 버튼 초기화

    const tabContentsContainer = document.querySelector('.inner-tab-contents');
    tabContentsContainer.innerHTML = ''; // 기존 탭 콘텐츠 초기화

    if (typeof recommendations === 'string') {
        try {
            recommendations = JSON.parse(recommendations); // 문자열이면 JSON으로 파싱
        } catch (error) {
            console.error('추천 정보를 파싱하는 데 실패했습니다:', error);
            addMessage('추천 정보를 불러오는데 실패했습니다.', 'bot-message');
            return;
        }
    }

    recommendations.forEach((candidate, index) => {
        // 각 후보자를 별도의 탭으로 생성
        const tabId = `candidate${index + 1}`;
        const tabButton = document.createElement('button');
        tabButton.className = 'inner-tab-button';
        tabButton.textContent = `후보자 ${index + 1}`;
        tabButton.onclick = (event) => openInnerTab(event, tabId);
        tabButtonsContainer.appendChild(tabButton);

        // 경력 정보 HTML로 변환
        let experienceHtml = '';
        if (candidate['경력']) {
            if (Array.isArray(candidate['경력'])) {
                candidate['경력'].forEach(exp => {
                    if (typeof exp === 'object') {
                        // exp의 구조에 따라 처리
                        if (exp['period']) {
                            // 'period' 키가 있는 경우
                            experienceHtml += `
                                <div class="experience-item" style="margin-bottom: 5px;">
                                    <p><strong>기간:</strong> ${exp['period']}</p>
                                    ${Object.keys(exp).filter(key => key !== 'period').map(key => `<p><strong>${key}:</strong> ${exp[key]}</p>`).join('')}
                                </div>
                            `;
                        } else {
                            // 기간이 키인 경우
                            for (let period in exp) {
                                if (exp.hasOwnProperty(period)) {
                                    const details = exp[period];
                                    experienceHtml += `
                                        <div class="experience-item" style="margin-bottom: 5px;">
                                            <p><strong>기간:</strong> ${period}</p>
                                            ${typeof details === 'object'
                                                ? Object.keys(details).map(key => `<p><strong>${key}:</strong> ${details[key]}</p>`).join('')
                                                : `<p>${details}</p>`
                                            }
                                        </div>
                                    `;
                                }
                            }
                        }
                    } else if (typeof exp === 'string') {
                        experienceHtml += `<p>${exp}</p>`;
                    } else {
                        console.error(`후보자 ${candidate['이름']}의 경력 형식이 예상과 다릅니다:`, exp);
                    }
                });
            } else if (typeof candidate['경력'] === 'object') {
                // 객체인 경우 처리
                for (let period in candidate['경력']) {
                    if (candidate['경력'].hasOwnProperty(period)) {
                        const details = candidate['경력'][period];
                        experienceHtml += `
                            <div class="experience-item" style="margin-bottom: 5px;">
                                <p><strong>기간:</strong> ${period}</p>
                                ${typeof details === 'object'
                                    ? Object.keys(details).map(key => `<p><strong>${key}:</strong> ${details[key]}</p>`).join('')
                                    : `<p>${details}</p>`
                                }
                            </div>
                        `;
                    }
                }
            } else if (typeof candidate['경력'] === 'string') {
                // 문자열인 경우 처리
                experienceHtml = `<p>${candidate['경력']}</p>`;
            } else {
                experienceHtml = '<p>경력 정보가 없습니다.</p>';
            }
        } else {
            experienceHtml = '<p>경력 정보가 없습니다.</p>';
        }

        // 후보자 정보를 HTML로 표시
        const candidateHtml = `
            <div id="${tabId}" class="inner-tab-content" style="display: none;">
                <div class="candidate-card" style="margin-bottom: 5px; padding: 10px; border: 1px solid #ccc;">
                    <h3 style="margin-bottom: 5px;">후보자 ${index + 1}: ${candidate['이름'] || 'N/A'}</h3>
                    <p style="margin: 0;"><strong>대학교:</strong> ${candidate['대학교'] || 'N/A'}</p>
                    <p style="margin: 0;"><strong>전공:</strong> ${candidate['전공'] || 'N/A'}</p>
                    <div class="experience-container" style="margin-top: 5px;">
                        <h4 style="margin: 5px 0;">경력:</h4>
                        ${experienceHtml}
                    </div>
                    <p style="margin: 5px 0;"><strong>종합 점수:</strong> ${candidate['종합 점수'] || 'N/A'}</p>
                    <canvas id="chart-candidate${index + 1}" width="300" height="150" style="max-width: 100%; margin-top: 10px;"></canvas>
                </div>
            </div>
        `;

        // 후보자 콘텐츠 추가
        tabContentsContainer.innerHTML += candidateHtml;

        // 개별 점수를 가져와서 차트에 사용
        const ageScore = parseFloat(candidate['Age_score']) || 0;
        const majorScore = parseFloat(candidate['Major_score']) || 0;
        const skillScore = parseFloat(candidate['Skill_score']) || 0;
        const experienceScore = parseFloat(candidate['Experience_score']) || 0;
        const RetrievalScore = parseFloat(candidate['리트리버 점수']) || 0;

        setTimeout(() => updateCandidateChart(`chart-candidate${index + 1}`, ageScore, majorScore, skillScore, experienceScore, RetrievalScore), 0);
    });

    // 첫 번째 탭 활성화
    if (tabButtonsContainer.children.length > 0) {
        tabButtonsContainer.children[0].classList.add('inner-active');
    }
    if (tabContentsContainer.children.length > 0) {
        tabContentsContainer.children[0].classList.add('inner-active');
        tabContentsContainer.children[0].style.display = 'block';
    }
}

function updateCandidateChart(chartId, ageScore, majorScore, skillScore, experienceScore, RetrievalScore) {
    const canvas = document.getElementById(chartId);
    if (!canvas) {
        console.error(`Canvas with id ${chartId} not found.`);
        return;
    }
    const ctx = canvas.getContext('2d');
    const labels = ['Age', 'Major', 'Skills', 'Experience', 'Retrieval'];

    const data = [
        ageScore,
        majorScore,
        skillScore,
        experienceScore,
        RetrievalScore,
    ];

    // Chart.js를 이용해 막대 차트를 생성합니다.
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Candidate Scores',
                data: data,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true, //기존 코드
            // maintainAspectRatio: false, //지피티 코드
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}


function openInnerTab(evt, tabId) {
    const tabContents = document.querySelectorAll('.inner-tab-content');
    tabContents.forEach(content => {
        content.classList.remove('inner-active');
        content.style.display = 'none';
    });

    const tabButtons = document.querySelectorAll('.inner-tab-button');
    tabButtons.forEach(button => {
        button.classList.remove('inner-active');
    });

    document.getElementById(tabId).classList.add('inner-active');
    document.getElementById(tabId).style.display = 'block';
    evt.currentTarget.classList.add('inner-active');
}