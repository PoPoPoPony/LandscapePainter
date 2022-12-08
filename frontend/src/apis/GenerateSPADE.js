// import request from '../../../utils/requestBackend'
import request from '@/utils/requestBackend'


export function GenerateSPADE(rgbArray) {
    var data = JSON.stringify(rgbArray)
    let config = {
        url: '/GenerateSPADE',
        method: 'POST',
        headers: {
            "Content-Type": "application/json"
        },
        data: data
    }
    

    return request(config)
}