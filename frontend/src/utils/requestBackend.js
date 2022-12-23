import axios from 'axios'


// axios.defaults.withCredentials = true
// axios.defaults.headers.get['Content-Type'] = 'application/x-www-form-urlencoded'

axios.defaults.retry = 3
axios.defaults.retryDelay = 10000;

const axios_instance = axios.create({
    // baseURL: "http://ponyia.ddns.net:8080/api/v1",
    baseURL: "http://localhost:8084/api/v1",
    timeout: 5000,
})



axios_instance.interceptors.response.use((res)=>{
    return res
}, (error)=>{
    var config = error.config
    config.retry=3
    config.retryDelay=2000;

    if(!config || !config.retry) return Promise.reject(error)
    config.__retryCount = config.__retryCount || 0
    if(config.__retryCount>=config.retry) {
        return Promise.reject(error)
    }
    config.__retryCount +=1
    var backoff = new Promise(function (resolve) {
        setTimeout(function() {
            resolve()
        }, config.retryDelay || 1)
    })
    return backoff.then(function() {
        return axios(config)
    })
})


export default axios_instance